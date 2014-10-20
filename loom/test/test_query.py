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

from itertools import izip
from nose.tools import assert_true, assert_equal, assert_not_equal
from distributions.dbg.random import sample_bernoulli
from distributions.io.stream import open_compressed
from distributions.fileutil import tempdir
from loom.schema_pb2 import ProductValue, CrossCat, Query
from loom.test.util import for_each_dataset
import loom.query
from loom.query import protobuf_to_data_row
import loom.config
from loom.test.util import load_rows

NONE = ProductValue.Observed.NONE
DENSE = ProductValue.Observed.DENSE


def set_observed(observed, observed_dense):
    observed.sparsity = DENSE
    observed.dense[:] = observed_dense


def set_diff(diff, observed_dense):
    diff.neg.observed.sparsity = NONE
    set_observed(diff.pos.observed, observed_dense)


def get_example_requests(model, rows, query_type='mixed'):
    assert query_type in ['sample', 'score', 'mixed']
    cross_cat = CrossCat()
    with open_compressed(model, 'rb') as f:
        cross_cat.ParseFromString(f.read())
    feature_count = sum(len(kind.featureids) for kind in cross_cat.kinds)
    featureids = range(feature_count)

    nontrivials = [True] * feature_count
    for kind in cross_cat.kinds:
        fs = iter(kind.featureids)
        for model in loom.schema.MODELS.iterkeys():
            for shared in getattr(kind.product_model, model):
                f = fs.next()
                if model == 'dd':
                    if len(shared.alphas) == 0:
                        nontrivials[f] = False
                elif model == 'dpd':
                    if len(shared.betas) == 0:
                        nontrivials[f] = False
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
        if query_type in ['sample', 'mixed']:
            set_diff(request.sample.data, none_observed)
            request.sample.to_sample.sparsity = DENSE
            request.sample.to_sample.dense[:] = observed
            request.sample.sample_count = 1
        if query_type in ['score', 'mixed']:
            set_diff(request.score.data, none_observed)
        requests.append(request)
    for row in load_rows(rows)[:20]:
        i += 1
        request = Query.Request()
        request.id = "example-{}".format(i)
        if query_type in ['sample', 'mixed']:
            request.sample.sample_count = 1
            request.sample.data.MergeFrom(row.diff)
            request.sample.to_sample.sparsity = DENSE
            conditions = izip(nontrivials, row.diff.pos.observed.dense)
            to_sample = [
                nontrivial and not is_observed
                for nontrivial, is_observed in conditions
            ]
            set_observed(request.sample.to_sample, to_sample)
        if query_type in ['score', 'mixed']:
            request.score.data.MergeFrom(row.diff)
        requests.append(request)
    return requests


def check_response(request, response):
    assert_equal(request.id, response.id)
    assert_equal(len(response.error), 0)


def get_response(server, request):
    server.send(request)
    return server.receive()


def _test_server(root, requests):
    with loom.query.ProtobufServer(root, debug=True) as protobuf_server:
        server = loom.query.QueryServer(protobuf_server)
        for request in requests:
            response = get_response(protobuf_server, request)
            check_response(request, response)
            if request.HasField('sample'):
                assert_equal(len(response.sample.samples), 1)
                pod_request = protobuf_to_data_row(request.sample.data)
                to_sample = request.sample.to_sample.dense[:]
                server.sample(to_sample, pod_request)
            if request.HasField('score'):
                assert_true(isinstance(response.score.score, float))
                pod_request = protobuf_to_data_row(request.score.data)
                server.score(pod_request)


@for_each_dataset
def test_sample(root, model, rows, **unused):
    requests = get_example_requests(model, rows, 'sample')
    _test_server(root, requests)


@for_each_dataset
def test_score(root, model, rows, **unused):
    requests = get_example_requests(model, rows, 'score')
    _test_server(root, requests)


@for_each_dataset
def test_batch_score(root, model, rows, **unused):
    requests = get_example_requests(model, rows, 'score')
    with loom.query.get_server(root, debug=True) as server:
        rows = [
            protobuf_to_data_row(request.score.data)
            for request in requests
        ]
        scores = list(server.batch_score(rows))
        assert_equal(len(scores), len(rows))


@for_each_dataset
def test_score_derivative_can_run(root, rows, **unused):
    with loom.query.get_server(root, debug=True) as server:
        rows = load_rows(rows)
        row = protobuf_to_data_row(rows[0].diff)
        diffs = server.score_derivative(row)
        assert len(rows) == len(diffs)
        diffs = server.score_derivative(row, row_limit=1)
        assert len(diffs) == 1


@for_each_dataset
def test_seed(root, model, rows, **unused):
    requests = get_example_requests(model, rows, 'mixed')
    with tempdir():
        loom.config.config_dump({'seed': 0}, 'config.pb.gz')
        with loom.query.ProtobufServer(root, config='config.pb.gz') as server:
            responses1 = [get_response(server, req) for req in requests]

    with tempdir():
        loom.config.config_dump({'seed': 0}, 'config.pb.gz')
        with loom.query.ProtobufServer(root, config='config.pb.gz') as server:
            responses2 = [get_response(server, req) for req in requests]

    with tempdir():
        loom.config.config_dump({'seed': 10}, 'config.pb.gz')
        with loom.query.ProtobufServer(root, config='config.pb.gz') as server:
            responses3 = [get_response(server, req) for req in requests]

    assert_equal(responses1, responses2)
    assert_not_equal(responses1, responses3)
