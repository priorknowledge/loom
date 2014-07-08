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
from nose.tools import assert_true, assert_equal
from distributions.dbg.random import sample_bernoulli
from distributions.fileutil import tempdir
from distributions.io.stream import open_compressed
from loom.schema_pb2 import ProductValue, CrossCat, Query
from loom.test.util import for_each_dataset, CLEANUP_ON_ERROR
import loom.query
from loom.query import SingleSampleProtobufServer, MultiSampleProtobufServer
from loom.query import protobuf_to_data_row
import loom.preql

CONFIG = {}


def get_example_requests(model, query_type='sample'):
    assert query_type in ['sample', 'score']
    cross_cat = CrossCat()
    with open_compressed(model, 'rb') as f:
        cross_cat.ParseFromString(f.read())
    feature_count = sum(len(kind.featureids) for kind in cross_cat.kinds)
    featureids = range(feature_count)

    nontrivials = [True] * feature_count
    for kind in cross_cat.kinds:
        fs = iter(kind.featureids)
        for model in loom.schema.MODELS.iterkeys():
            for f, shared in izip(fs, getattr(kind.product_model, model)):
                if model == 'dd':
                    nontrivials[f] = (len(shared.alphas) > 0)
                elif model == 'dpd':
                    nontrivials[f] = (len(shared.betas) > 0)

    all_observed = nontrivials[:]
    none_observed = [False] * feature_count
    observeds = []
    observeds.append(all_observed)
    for f, nontrivial in izip(featureids, nontrivials):
        if nontrivial:
            observed = all_observed[:]
            observed[f] = False
            observeds.append(observed)
    for f in featureids:
        observed = [
            nontrivial and sample_bernoulli(0.5)
            for nontrivial in nontrivials
        ]
        observeds.append(observed)
    for f, nontrivial in izip(featureids, nontrivials):
        if nontrivial:
            observed = none_observed[:]
            observed[f] = True
            observeds.append(observed)
    observeds.append(none_observed)

    requests = []
    for i, observed in enumerate(observeds):
        request = Query.Request()
        request.id = "example-{}".format(i)
        if query_type == 'sample':
            request.sample.data.observed.sparsity = ProductValue.Observed.DENSE
            request.sample.data.observed.dense[:] = none_observed
            request.sample.to_sample.sparsity = ProductValue.Observed.DENSE
            request.sample.to_sample.dense[:] = observed
            request.sample.sample_count = 1
        elif query_type == 'score':
            request.score.data.observed.sparsity = ProductValue.Observed.DENSE
            request.score.data.observed.dense[:] = none_observed
        requests.append(request)
    return requests


def get_server(model, groups, server_module):
    config_in = os.path.abspath('config.pb.gz')
    loom.config.config_dump(CONFIG, config_in)
    kwargs = {
        'config_in': config_in,
        'model_in': model,
        'groups_in': groups,
        'debug': True,
    }
    return server_module(**kwargs)


def check_response(request, response):
    assert_equal(request.id, response.id)
    assert_equal(len(response.error), 0)


def get_response(server, request):
    server.send(request)
    return server.receive()


@for_each_dataset
def test_sample_one(model, groups, **unused):
    requests = get_example_requests(model, 'sample')
    with tempdir(cleanup_on_error=CLEANUP_ON_ERROR):
        args = [model, groups, SingleSampleProtobufServer]
        with get_server(*args) as server:
                query_server = loom.query.QueryServer(server)
                for request in requests:
                    response = get_response(server, request)
                    check_response(request, response)
                    assert_equal(len(response.sample.samples), 1)
                    pod_request = protobuf_to_data_row(request.sample.data)
                    to_sample = request.sample.to_sample.dense[:]
                    query_server.sample(to_sample, pod_request)


@for_each_dataset
def test_sample_multi(model, groups, **unused):
    requests = get_example_requests(model, 'sample')
    with tempdir(cleanup_on_error=CLEANUP_ON_ERROR):
        args = [[model, model], [groups, groups], MultiSampleProtobufServer]
        with get_server(*args) as server:
                query_server = loom.query.QueryServer(server)
                for request in requests:
                    response = get_response(server, request)
                    check_response(request, response)
                    assert_equal(len(response.sample.samples), 1)
                    pod_request = protobuf_to_data_row(request.sample.data)
                    to_sample = request.sample.to_sample.dense[:]
                    query_server.sample(to_sample, pod_request)


@for_each_dataset
def test_score_one(model, groups, **unused):
    requests = get_example_requests(model, 'score')
    with tempdir(cleanup_on_error=CLEANUP_ON_ERROR):
        args = [model, groups, SingleSampleProtobufServer]
        with get_server(*args) as server:
                query_server = loom.query.QueryServer(server)
                for request in requests:
                    response = get_response(server, request)
                    check_response(request, response)
                    assert_true(isinstance(response.score.score, float))
                    pod_request = protobuf_to_data_row(request.score.data)
                    query_server.score(pod_request)


@for_each_dataset
def test_score_multi(model, groups, **unused):
    requests = get_example_requests(model, 'score')
    with tempdir(cleanup_on_error=CLEANUP_ON_ERROR):
        args = [[model, model], [groups, groups], MultiSampleProtobufServer]
        with get_server(*args) as server:
                query_server = loom.query.QueryServer(server)
                for request in requests:
                    response = get_response(server, request)
                    check_response(request, response)
                    assert_true(isinstance(response.score.score, float))
                    pod_request = protobuf_to_data_row(request.score.data)
                    query_server.score(pod_request)
