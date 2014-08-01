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

from itertools import product
import numpy
from nose import SkipTest
from nose.tools import (
    assert_greater,
    assert_almost_equal,
)
from distributions.util import (
    density_goodness_of_fit,
    discrete_goodness_of_fit,
)
import loom.preql
import loom.query
from loom.query import (
    SingleSampleProtobufServer,
    MultiSampleProtobufServer
)
from loom.test.util import for_each_dataset, load_rows


GOF_EXP = 3
MIN_GOODNESS_OF_FIT = 10. ** (-GOF_EXP)

SAMPLE_COUNT = 300

# tests are inaccurate with highly imbalanced data
MIN_CATEGORICAL_PROB = .03


@for_each_dataset
def test_score_none(samples, encoding, **unused):
    cases = [
        (SingleSampleProtobufServer, samples[0]),
        (MultiSampleProtobufServer, samples[:1]),
        (MultiSampleProtobufServer, samples),
        (MultiSampleProtobufServer, [samples[0], samples[0]]),
    ]
    for Server, samples in cases:
        with Server(samples, debug=True) as protobuf_server:
            query_server = loom.query.QueryServer(protobuf_server)
            preql = loom.preql.PreQL(query_server, encoding)
            fnames = preql.feature_names
            assert_almost_equal(
                query_server.score([None for _ in fnames]),
                0.,
                places=GOF_EXP)


@for_each_dataset
def test_mi_entropy_relations(samples, encoding, **unused):
    with loom.query.get_server(samples, debug=True) as query_server:
        preql = loom.preql.PreQL(query_server, encoding)
        fnames = preql.feature_names
        feature_sets = [
            [fnames[0]],
            [fnames[2]],
            [fnames[0], fnames[1]],
        ]
        to_sample = preql.cols_to_bools([fnames[i] for i in [0, 1, 2]])
        samples = query_server.sample(to_sample, sample_count=10)
        for fset1, fset2 in product(feature_sets, feature_sets):
            to_score1 = preql.cols_to_bools(fset1)
            to_score2 = preql.cols_to_bools(fset2)
            to_score = preql.cols_to_bools(fset1 + fset2)
            mutual_info = query_server.mutual_information(
                samples,
                to_score1,
                to_score2)
            entropy1 = query_server.entropy(samples, to_score1)
            entropy2 = query_server.entropy(samples, to_score2)
            entropy_joint = query_server.entropy(samples, to_score)
            if to_score1 == to_score2:
                measures = [mutual_info, entropy1, entropy2, entropy_joint]
                for m1, m2 in product(measures, measures):
                    assert_almost_equal(m1.mean, m2.mean)
            expected = mutual_info.mean
            actual = entropy1.mean + entropy2.mean - entropy_joint.mean
            assert_almost_equal(actual, expected)


def _check_marginal_samples_match_scores(protobuf_server, row, fi):
    query_server = loom.query.QueryServer(protobuf_server)
    row = loom.query.protobuf_to_data_row(row.diff)
    row[fi] = None
    to_sample = [i == fi for i in range(len(row))]
    samples = query_server.sample(to_sample, row, SAMPLE_COUNT)
    val = samples[0][fi]
    base_score = query_server.score(row)
    if isinstance(val, bool) or isinstance(val, int):
        probs_dict = {}
        samples = [sample[fi] for sample in samples]
        for sample in set(samples):
            row[fi] = sample
            probs_dict[sample] = numpy.exp(
                query_server.score(row) - base_score)
        if len(probs_dict) == 1:
            assert_almost_equal(probs_dict[sample], 1., places=GOF_EXP)
            return
        if min(probs_dict.values()) < MIN_CATEGORICAL_PROB:
            return
        gof = discrete_goodness_of_fit(samples, probs_dict, plot=True)
    elif isinstance(val, float):
        probs = numpy.exp([
            query_server.score(sample) - base_score
            for sample in samples
        ])
        samples = [sample[fi] for sample in samples]
        gof = density_goodness_of_fit(samples, probs, plot=True)
    assert_greater(gof, MIN_GOODNESS_OF_FIT)


@for_each_dataset
def test_samples_match_scores_one(samples, rows, **unused):
    raise SkipTest('FIXME(jglidden) test fails too often')
    Server = SingleSampleProtobufServer
    rows = load_rows(rows)
    rows = rows[::len(rows) / 2]
    with Server(samples[0], debug=True) as protobuf_server:
        for row in rows:
            _check_marginal_samples_match_scores(protobuf_server, row, 0)


@for_each_dataset
def test_samples_match_scores_multi(samples, rows, **unused):
    Server = MultiSampleProtobufServer
    rows = load_rows(rows)
    rows = rows[::len(rows) / 2]
    with Server(samples, debug=True) as protobuf_server:
        for row in rows:
            _check_marginal_samples_match_scores(protobuf_server, row, 0)
