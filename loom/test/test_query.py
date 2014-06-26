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
from itertools import izip
from nose.tools import assert_false, assert_true, assert_equal
from distributions.dbg.random import sample_bernoulli
from distributions.fileutil import tempdir
from distributions.io.stream import open_compressed
from loom.schema_pb2 import CrossCat, Query
from loom.test.util import for_each_dataset, CLEANUP_ON_ERROR
import loom.query

CONFIG = {}


def get_example_requests(model):
    cross_cat = CrossCat()
    with open_compressed(model) as f:
        cross_cat.ParseFromString(f.read())
    feature_count = sum(len(kind.featureids) for kind in cross_cat.kinds)

    all_observed = [True] * feature_count
    none_observed = [False] * feature_count
    observeds = []
    observeds.append(all_observed)
    for f in xrange(feature_count):
        observed = all_observed[:]
        observed[f] = False
        observeds.append(observed)
    for f in xrange(feature_count):
        observed = [sample_bernoulli(0.5) for _ in xrange(feature_count)]
        observeds.append(observed)
    for f in xrange(feature_count):
        observed = none_observed[:]
        observed[f] = True
        observeds.append(observed)
    observeds.append(none_observed)

    requests = []
    for i, observed in enumerate(observeds):
        request = Query.Request()
        request.id = "example-{}".format(i)
        request.sample.data.observed[:] = none_observed
        request.sample.to_sample[:] = observed
        request.sample.sample_count = 1
        requests.append(request)

    return requests


@for_each_dataset
def test_server(model, groups, **unused):
    requests = get_example_requests(model)
    with tempdir(cleanup_on_error=CLEANUP_ON_ERROR):
        config_in = os.path.abspath('config.pb.gz')
        loom.config.config_dump(CONFIG, config_in)
        kwargs = {
            'config_in': config_in,
            'model_in': model,
            'groups_in': groups,
            'debug': True,
        }
        with loom.query.serve(**kwargs) as server:
            responses = [server.call_protobuf(request) for request in requests]

    with tempdir(cleanup_on_error=CLEANUP_ON_ERROR):
        config_in = os.path.abspath('config.pb.gz')
        loom.config.config_dump(CONFIG, config_in)
        kwargs = {
            'config_in': config_in,
            'model_in': model,
            'groups_in': groups,
            'debug': True,
        }
        with loom.query.serve(**kwargs) as server:
            for request in requests:
                req = Query.Request()
                req.id = request.id
                req.score.data.observed[:] = request.sample.data.observed[:]
                res = server.call_protobuf(req)
                assert_equal(req.id, res.id)
                assert_false(hasattr(req, 'error'))
                assert_true(isinstance(res.score.score, float))

    for request, response in izip(requests, responses):
        assert_equal(request.id, response.id)
        assert_false(hasattr(request, 'error'))
        assert_equal(len(response.sample.samples), 1)


@for_each_dataset
def test_batch_predict(model, groups, **unused):
    requests = get_example_requests(model)
    with tempdir(cleanup_on_error=CLEANUP_ON_ERROR):
        config_in = os.path.abspath('config.pb.gz')
        loom.config.config_dump(CONFIG, config_in)
        responses = loom.query.batch_predict(
            config_in=config_in,
            model_in=model,
            groups_in=groups,
            requests=requests,
            debug=True)
    assert_equal(len(responses), len(requests))
    for request, response in izip(requests, responses):
        assert_equal(request.id, response.id)
        assert_false(hasattr(request, 'error'))
        assert_equal(len(response.sample.samples), 1)
