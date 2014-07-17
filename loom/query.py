# Copyright (c) 2014, Salesforce.com, Inc.  All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# - Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
# - Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
# - Neither the name of Salesforce.com nor the names of its contributors
#   may be used to endorse or promote products derived from this
#   software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
# OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
# TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
# USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import loom.runner
from distributions.io.stream import protobuf_stream_write, protobuf_stream_read
from distributions.lp.random import log_sum_exp
from loom.schema_pb2 import Query, ProductValue
import loom.cFormat
import numpy as np
from copy import copy
from itertools import chain
import uuid


def even_unif_multinomial(total_count, num_choices):
    '''
    This is a lower-variance approximation to a uniform multinomial sampler
    which offers better load balancing and better downstream point estimates.
    The resulting predictions will still be exchangeable, but not independent.
    As a benefit, any MC estimator based on these predictions will have lower
    variance than an estimator using iid multinomial samples.
    '''
    quotient = int(total_count / num_choices)
    remainder = total_count - quotient * num_choices
    result = np.ones((num_choices,), dtype=int) * quotient
    result[:remainder] += 1
    assert result.sum() == total_count
    result = result.tolist()
    np.random.shuffle(result)
    return result


def split_by_type(data_row):
    booleans = []
    counts = []
    reals = []
    mask = []
    for val in data_row:
        if val is not None:
            mask.append(True)
            if isinstance(val, bool):
                booleans.append(val)
            elif isinstance(val, int):
                counts.append(val)
            elif isinstance(val, float):
                reals.append(val)
        else:
            mask.append(False)
    return mask, booleans, counts, reals


def data_row_to_protobuf(data_row, message):
    assert isinstance(message, ProductValue)
    mask, booleans, counts, reals = split_by_type(data_row)
    message.observed.dense[:] = mask
    message.booleans[:] = booleans
    message.counts[:] = counts
    message.reals[:] = reals


def protobuf_to_data_row(message):
    assert isinstance(message, ProductValue)
    mask = message.observed.dense[:]
    data_row = []
    vals = chain(
        message.booleans,
        message.counts,
        message.reals)
    for marker in mask:
        if marker:
            data_row.append(vals.next())
        else:
            data_row.append(None)
    return data_row


def load_data_rows(filename):
    for row in loom.cFormat.row_stream_load(filename):
        data = row.iter_data()
        packed = chain(data['booleans'], data['counts'], data['reals'])
        yield [
            packed.next() if observed else None
            for observed in data['observed']
        ]


class QueryServer(object):
    def __init__(self, protobuf_server):
        self.protobuf_server = protobuf_server

    def close(self):
        self.protobuf_server.close()

    def __enter__(self):
        return self

    def __exit__(self, *unused):
        self.close()

    def request(self):
        request = Query.Request()
        request.id = str(uuid.uuid4())
        return request

    def sample(self, to_sample, conditioning_row=None, sample_count=10):
        if conditioning_row is None:
            conditioning_row = [None for _ in to_sample]
        assert len(to_sample) == len(conditioning_row)
        request = self.request()
        request.sample.data.observed.sparsity = ProductValue.Observed.DENSE
        data_row_to_protobuf(
            conditioning_row,
            request.sample.data)
        request.sample.to_sample.sparsity = ProductValue.Observed.DENSE
        request.sample.to_sample.dense[:] = to_sample
        request.sample.sample_count = sample_count
        self.protobuf_server.send(request)
        response = self.protobuf_server.receive()
        if response.error:
            raise Exception('\n'.join(response.error))
        samples = []
        for sample in response.sample.samples:
            data_out = protobuf_to_data_row(sample)
            for i, val in enumerate(data_out):
                if val is None:
                    assert to_sample[i] is False
                    data_out[i] = conditioning_row[i]
            samples.append(data_out)
        return samples

    def score(self, row):
        request = self.request()
        request.score.data.observed.sparsity = ProductValue.Observed.DENSE
        data_row_to_protobuf(
            row,
            request.score.data)
        self.protobuf_server.send(request)
        response = self.protobuf_server.receive()
        if response.error:
            raise Exception('\n'.join(response.error))
        return response.score.score

    def entropy(self, to_sample, conditioning_row=None, sample_count=100):
        '''
        Estimate the entropy
        '''
        if conditioning_row is not None:
            offest = self.score(conditioning_row)
        else:
            offset = 0.
        samples = self.sample(to_sample, conditioning_row, sample_count)
        entropys = np.array([-self.score(sample) for sample in samples])
        entropys -= offset
        entropy_estimate = np.mean(entropys)
        error_estimate = np.sqrt(np.var(entropys)/sample_count)
        return entropy_estimate, error_estimate

    def mutual_information(
            self,
            to_sample1,
            to_sample2,
            conditioning_row=None,
            sample_count=1000):
        '''
        Estimate the mutual information between columns1 and columns2
        conditioned on conditioning_row
        '''
        if conditioning_row is None:
            conditioning_row = [None for _ in to_sample1]
            offset = 0.
        else:
            offset = self.score(conditioning_row)
        assert len(to_sample1) == len(to_sample2)
        to_sample = [(a or b) for a, b in zip(to_sample1, to_sample2)]
        assert len(to_sample) == len(conditioning_row)

        samples = self.sample(to_sample, conditioning_row, sample_count)

        def comp_row(to_sample, sample, conditioning_row):
            row = []
            for ts, val, cond_val in zip(to_sample, sample, conditioning_row):
                if ts is True:
                    row.append(val)
                else:
                    row.append(cond_val)
            return row

        mis = np.zeros(sample_count)
        for i, sample in enumerate(samples):
            mis[i] += self.score(comp_row(to_sample, sample, conditioning_row))
            mis[i] -= self.score(comp_row(to_sample1, sample, conditioning_row))
            mis[i] -= self.score(comp_row(to_sample2, sample, conditioning_row))
        mis += offset
        mi_estimate = np.mean(mis)
        error_estimate = np.sqrt(np.var(mis)/sample_count)
        return mi_estimate, error_estimate


class MultiSampleProtobufServer(object):
    def __init__(self, **kwargs):
        self.servers = []
        model_ins = kwargs['model_in']
        groups_ins = kwargs['groups_in']
        assert isinstance(model_ins, list)
        assert isinstance(groups_ins, list)
        for model_in, groups_in in zip(model_ins, groups_ins):
            kwargs_one = copy(kwargs)
            kwargs_one['model_in'] = model_in
            kwargs_one['groups_in'] = groups_in
            single_server = SingleSampleProtobufServer(**kwargs_one)
            self.servers.append(single_server)

    def send(self, request):
        requests = []
        for server in self.servers:
            req = Query.Request()
            req.CopyFrom(request)
            requests.append(req)
        if request.HasField("sample"):
            total_count = request.sample.sample_count
            per_server_counts = even_unif_multinomial(
                total_count,
                len(self.servers))
            # TODO handle 0 counts?
            for req, count in zip(requests, per_server_counts):
                req.sample.sample_count = count
        if request.HasField("score"):
            # score requests passed to each sample
            pass
        for req, server in zip(requests, self.servers):
            server.send(req)

    def receive(self):
        responses = [server.receive() for server in self.servers]
        assert len(set([res.id for res in responses])) == 1

        samples = [res.sample.samples for res in responses]
        samples = list(chain(*samples))
        np.random.shuffle(samples)
        #FIXME what if request did not have score
        score_part = log_sum_exp([res.score.score for res in responses])
        score = score_part - np.log(len(responses))

        response = Query.Response()
        response.id = responses[0].id  # HACK
        for res in responses:
            response.error.extend(res.error)
        response.sample.samples.extend(samples)
        response.score.score = score
        return response

    def call(self, request):
        response = Query.Response()
        if request.HasField("sample"):
            self.__sample(request, response)
        if request.HasField("score"):
            self.__score(request, response)
        return response

    def close(self):
        for server in self.servers:
            server.close()

    def __enter__(self):
        return self

    def __exit__(self, *unused):
        self.close()


class SingleSampleProtobufServer(object):
    def __init__(
            self,
            config_in,
            model_in,
            groups_in,
            log_out=None,
            debug=False,
            profile=None):
        self.proc = loom.runner.query(
            config_in=config_in,
            model_in=model_in,
            groups_in=groups_in,
            log_out=log_out,
            debug=debug,
            profile=profile,
            block=False)

    def call_string(self, request_string):
        protobuf_stream_write(request_string, self.proc.stdin)

    def send(self, request):
        assert isinstance(request, Query.Request)
        request_string = request.SerializeToString()
        self.call_string(request_string)

    def receive(self):
        response_string = protobuf_stream_read(self.proc.stdout)
        response = Query.Response()
        response.ParseFromString(response_string)
        return response

    def close(self):
        self.proc.stdin.close()
        self.proc.wait()

    def __enter__(self):
        return self

    def __exit__(self, *unused):
        self.close()
