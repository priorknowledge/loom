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
from loom.schema_pb2 import Query, ProductValue
import loom.cFormat
import loom.runner


SAMPLE_COUNT = {
    'sample': 10,
    'entropy': 300,
    'mutual_information': 300
}
BUFFER_SIZE = 10

Estimate = namedtuple('Estimate', ['mean', 'variance'], verbose=False)


def get_estimate(samples):
    mean = numpy.mean(samples)
    variance = numpy.var(samples) / len(samples)
    return Estimate(mean, variance)


NONE = ProductValue.Observed.NONE
DENSE = ProductValue.Observed.DENSE


def data_row_to_protobuf(data_row, message):
    assert isinstance(message, ProductValue.Diff)
    message.Clear()
    message.neg.observed.sparsity = NONE
    message.pos.observed.sparsity = DENSE
    mask = message.pos.observed.dense
    fields = {
        bool: message.pos.booleans,
        int: message.pos.counts,
        float: message.pos.reals,
    }
    for val in data_row:
        observed = val is not None
        mask.append(observed)
        if observed:
            fields[type(val)].append(val)


def protobuf_to_data_row(diff):
    assert isinstance(diff, ProductValue.Diff)
    assert diff.neg.observed.sparsity == NONE
    data = diff.pos
    packed = chain(data.booleans, data.counts, data.reals)
    data_row = []
    for observed in data.observed.dense:
        data_row.append(packed.next() if observed else None)
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

    def _send_score(self, row):
        request = self.request()
        data_row_to_protobuf(row, request.score.data)
        self.protobuf_server.send(request)

    def _receive_score(self):
        response = self.protobuf_server.receive()
        if response.error:
            raise Exception('\n'.join(response.error))
        return response.score.score

    def score(self, row):
        self._send_score(row)
        return self._receive_score()

    def batch_score(self, rows, buffer_size=BUFFER_SIZE):
        buffered = 0
        for row in rows:
            self._send_score(row)
            if buffered < buffer_size:
                buffered += 1
            else:
                yield self._receive_score()
        for _ in xrange(buffered):
            yield self._receive_score()

    def _entropy_from_samples(self, variable_masks, samples, conditioning_row):
        if conditioning_row is not None:
            base_score = self.score(conditioning_row)
        else:
            base_score = 0.

        def restrict(row, variable_mask):
            return [val if m else None for val, m in izip(row, variable_mask)]

        results = {}
        # this has complexity O(sample_count * len(variable_masks) ** 2)
        for vm in variable_masks:
            scores = numpy.array(list(
                self.batch_score(restrict(sample, vm) for sample in samples)))
            results[vm] = get_estimate(base_score - scores)
        return results

    def entropy(
            self,
            variable_masks,
            conditioning_row=None,
            sample_count=None):
        '''
        Estimate the entropy for each column group in variable_masks
        '''
        if sample_count is None:
            sample_count = SAMPLE_COUNT['entropy']
        joint_mask = [any(flags) for flags in izip(*variable_masks)]
        samples = self.sample(joint_mask, conditioning_row, sample_count)
        return self._entropy_from_samples(
            variable_masks,
            samples,
            conditioning_row)

    def mutual_information(
            self,
            variable_mask1,
            variable_mask2,
            entropys=None,
            conditioning_row=None,
            sample_count=None):
        '''
        Estimate the mutual information between columns1 and columns2
        conditioned on conditioning_row
        '''
        if sample_count is None:
            sample_count = SAMPLE_COUNT['mutual_information']
        if conditioning_row is None:
            conditioning_row = [None for _ in variable_mask1]
        assert len(variable_mask1) == len(variable_mask2)
        variable_mask = tuple(
            [(a or b) for a, b in izip(variable_mask1, variable_mask2)])
        assert len(variable_mask) == len(conditioning_row)

        if entropys is None:
            entropys = self.entropy(
                [variable_mask1, variable_mask1, variable_mask],
                conditioning_row,
                sample_count)
        mi = entropys[variable_mask1].mean \
            + entropys[variable_mask2].mean \
            - entropys[variable_mask].mean
        variance = entropys[variable_mask1].variance \
            + entropys[variable_mask2].variance \
            + entropys[variable_mask].variance
        return Estimate(mi, variance)


class ProtobufServer(object):
    def __init__(self, root, debug=False, profile=None):
        self.proc = loom.runner.query(
            root_in=root,
            log_out=None,
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


def get_server(root, debug=False, profile=None):
    protobuf_server = ProtobufServer(root, debug, profile)
    return QueryServer(protobuf_server)
