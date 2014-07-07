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
from loom.schema_pb2 import Query
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


class QueryServer(object):
    def __init__(self, protobuf_server):
        self.protobuf_server = protobuf_server

    def close(self):
        self.protobuf_server.close()

    def __enter__(self):
        return self

    def __exit(self, etc):
        self.close()

    def request(self):
        request = Query.Request()
        request.id = str(uuid.uuid4())
        return request

    def sample(self, to_sample, conditioning_row=None, sample_count=10):
        request = self.request()
        request.sample.data = conditioning_row
        request.sample.to_sample = to_sample
        request.sample.count = sample_count
        self.protobuf_server.send(request)
        return self.protobuf_server.receive(request)

    def score(self, row, conditioning_row=None):
        request = self.request()
        request.score.data = conditioning_row
        self.protobuf_server.send(request)
        return self.protobuf_server.receive(request)


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
        score = np.logaddexp.reduce([res.score.score for res in responses])

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

    def __exit__(self, type, value, traceback):
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
        log_out = loom.runner.optional_file(log_out)
        command = [
            'query',
            config_in, model_in, groups_in, '-',
            '-', log_out,
        ]
        loom.runner.assert_found(config_in, model_in, groups_in)
        self.proc = loom.runner.popen_piped(command, debug)

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

    def __exit__(self, type, value, traceback):
        self.close()
