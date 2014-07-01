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

import os
import loom.runner
from distributions.fileutil import tempdir
from loom.schema_pb2 import Query
from numpy import logaddexp

serve = loom.runner.query.serve


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


class Server(Object):
    def __init__(self, samples_in, **kwargs):
        self.servers = [serve(**dict(sample_in, kwargs)) for sample_in in samples_in]

    def __sample(request, response):
        total_count = request.sample_count
        per_server_counts = even_unif_multinmoial(total_count, len(self.servers))
        sample_response = Query.Sample.Response()
        errors = []
        for server, count in zip(self.servers, per_server_counts):
            request_in = Query.Request.CopyFrom(request)
            request_in.sample.sample_count = count
            response_out = server.call_protobuf(request_in)
            if response_out.has_error():
                errors.append(response_out.error)
            else:
                sample_response.samples.extend(response_out.samples)
        if errors:
            response.error[:] = errors
        else:
            response.sample = sample_response

    def __score(request, response):
        score_response = Query.Score.Response()
        responses = [server(request) for server in self.servers]
        scores = []
        errors = []
        for response_out in responses:
            if response_out.has_error():
                errors.append(response_out.error)
            else:
                scores.append(response_out.score)
        score_response.score = logaddexp.reduce(scores)
        if errors:
            response.error[:] = errors
        else:
            request.score = score_response

    def __call__(request):
        response = Query.Response()
        response.id = request.id
        if request.has_sample():
            __sample(request, response)
        if request.has_score():
            __score(request, response)
        return response








