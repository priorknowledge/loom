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

import uuid
from itertools import izip, chain
from collections import namedtuple
import numpy
from distributions.io.stream import protobuf_stream_write, protobuf_stream_read
from distributions.lp.random import log_sum_exp
from loom.schema_pb2 import Query, ProductValue
import loom.cFormat
import loom.runner


SAMPLE_COUNT = {
    'sample': 10,
    'entropy': 300,
    'mutual_information': 300
}

Estimate = namedtuple('Estimate', ['mean', 'variance'], verbose=False)


def get_estimate(samples):
    mean = numpy.mean(samples)
    variance = numpy.var(samples) / len(samples)
    return Estimate(mean, variance)


NONE = ProductValue.Observed.NONE
DENSE = ProductValue.Observed.DENSE


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
    result = numpy.ones((num_choices,), dtype=int) * quotient
    result[:remainder] += 1
    assert result.sum() == total_count
    result = result.tolist()
    numpy.random.shuffle(result)
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
    assert isinstance(message, ProductValue.Diff)
    mask, booleans, counts, reals = split_by_type(data_row)
    message.Clear()
    message.neg.observed.sparsity = NONE
    message.pos.observed.sparsity = DENSE
    message.pos.observed.dense[:] = mask
    message.pos.booleans[:] = booleans
    message.pos.counts[:] = counts
    message.pos.reals[:] = reals


def protobuf_to_data_row(diff):
    assert isinstance(diff, ProductValue.Diff)
    assert diff.neg.observed.sparsity == NONE
    data = diff.pos
    packed = chain(data.booleans, data.counts, data.reals)
    data_row = []
    for marker in data.observed.dense:
        if marker:
            data_row.append(packed.next())
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

    def sample(self, to_sample, conditioning_row=None, sample_count=None):
        if sample_count is None:
            sample_count = SAMPLE_COUNT['sample']
        if conditioning_row is None:
            conditioning_row = [None for _ in to_sample]
        assert len(to_sample) == len(conditioning_row)
        request = self.request()
        data_row_to_protobuf(conditioning_row, request.sample.data)
        request.sample.to_sample.sparsity = DENSE
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
        data_row_to_protobuf(row, request.score.data)
        self.protobuf_server.send(request)
        response = self.protobuf_server.receive()
        if response.error:
            raise Exception('\n'.join(response.error))
        return response.score.score

    def _fill_conditions(self, observed, sample, conditioning_row):
        return [
            val if ts else cval
            for ts, val, cval in izip(observed, sample, conditioning_row)
        ]

    def entropy(self, samples, columns, conditioning_row=None):
        '''
        Estimate the entropy of samples with respect to columns
        '''
        if conditioning_row is None:
            conditioning_row = [None for _ in columns]
            base_score = 0.
        else:
            base_score = self.score(conditioning_row)
        samples = [self._fill_conditions(columns, sample, conditioning_row)
                   for sample in samples]
        entropys = numpy.array([
            base_score - self.score(sample)
            for sample in samples
        ])
        return get_estimate(entropys)

    def mutual_information(
            self,
            samples,
            columns1,
            columns2,
            conditioning_row=None):
        '''
        Estimate the mutual information between columns1 and columns2
        conditioned on conditioning_row
        with respect to samples
        '''
        if conditioning_row is None:
            conditioning_row = [None for _ in columns1]
            base_score = 0.
        else:
            base_score = self.score(conditioning_row)
        assert len(columns1) == len(columns2)
        columns_union = [(a or b) for a, b in izip(columns1, columns2)]
        assert len(columns_union) == len(conditioning_row)

        mis = numpy.zeros(len(samples))
        for i, sample in enumerate(samples):
            joint_row = self._fill_conditions(
                columns_union,
                sample,
                conditioning_row)
            mis[i] += self.score(joint_row)

            sample_row1 = self._fill_conditions(
                columns1,
                sample,
                conditioning_row)
            mis[i] -= self.score(sample_row1)

            sample_row2 = self._fill_conditions(
                columns2,
                sample,
                conditioning_row)
            mis[i] -= self.score(sample_row2)
        mis += base_score
        return get_estimate(mis)


class MultiSampleProtobufServer(object):
    def __init__(self, samples, debug=False, profile=None):
        self.servers = [
            SingleSampleProtobufServer(sample, debug, profile)
            for sample in samples
        ]

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
            for req, count in izip(requests, per_server_counts):
                req.sample.sample_count = count
        if request.HasField("score"):
            # score requests passed to each sample
            pass
        for req, server in izip(requests, self.servers):
            server.send(req)

    def receive(self):
        responses = [server.receive() for server in self.servers]
        assert len(set([res.id for res in responses])) == 1

        samples = [res.sample.samples for res in responses]
        samples = list(chain(*samples))
        numpy.random.shuffle(samples)
        # FIXME what if request did not have score
        score_part = log_sum_exp([res.score.score for res in responses])
        score = score_part - numpy.log(len(responses))

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
    def __init__(self, paths, debug=False, profile=None):
        self.proc = loom.runner.query(
            config_in=paths['config'],
            model_in=paths['model'],
            groups_in=paths['groups'],
            log_out=paths['query_log'],
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


def get_server(samples, debug=False, profile=None):
    assert isinstance(samples, list), samples
    protobuf_server = MultiSampleProtobufServer(samples, debug, profile)
    return QueryServer(protobuf_server)
