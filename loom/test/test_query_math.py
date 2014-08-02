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
    assert_almost_equal,
    assert_greater,
    assert_less,
)
from distributions.util import (
    density_goodness_of_fit,
    discrete_goodness_of_fit,
)
import loom.preql
import loom.query
from loom.test.util import for_each_dataset, load_rows


MIN_GOODNESS_OF_FIT = 1e-4
SCORE_TOLERANCE = 1e-3

SAMPLE_COUNT = 500

# tests are inaccurate with highly imbalanced data
MIN_CATEGORICAL_PROB = .03


@for_each_dataset
def test_score_none(root, encoding, **unused):
    with loom.query.get_server(root, debug=True) as server:
        preql = loom.preql.PreQL(server, encoding)
        fnames = preql.feature_names
        assert_less(
            abs(server.score([None for _ in fnames])),
            SCORE_TOLERANCE)


@for_each_dataset
def test_mi_entropy_relations(root, encoding, **unused):
    raise SkipTest('FIXME(jglidden) test fails too often')
    with loom.query.get_server(root, debug=True) as server:
        preql = loom.preql.PreQL(server, encoding)
        fnames = preql.feature_names
        feature_sets = [
            [fnames[0]],
            [fnames[2]],
            [fnames[0], fnames[1]],
        ]
        for fset1, fset2 in product(feature_sets, feature_sets):
            to_sample1 = preql.cols_to_sample(fset1)
            to_sample2 = preql.cols_to_sample(fset2)
            to_sample = preql.cols_to_sample(fset1 + fset2)
            mutual_info = server.mutual_information(
                to_sample1,
                to_sample2,
                sample_count=SAMPLE_COUNT)
            entropy1 = server.entropy(
                to_sample1,
                sample_count=SAMPLE_COUNT)
            entropy2 = server.entropy(
                to_sample2,
                sample_count=SAMPLE_COUNT)
            entropy_joint = server.entropy(
                to_sample,
                sample_count=SAMPLE_COUNT)
            if to_sample1 == to_sample2:
                measures = [mutual_info, entropy1, entropy2, entropy_joint]
                for m1, m2 in product(measures, measures):
                    assert_almost_equal(m1.mean, m2.mean)
            expected = mutual_info.mean
            actual = entropy1.mean + entropy2.mean - entropy_joint.mean
            assert_almost_equal(actual, expected)


def _check_marginal_samples_match_scores(server, row, fi):
    row = loom.query.protobuf_to_data_row(row.diff)
    row[fi] = None
    to_sample = [i == fi for i in range(len(row))]
    samples = server.sample(to_sample, row, SAMPLE_COUNT)
    val = samples[0][fi]
    base_score = server.score(row)
    if isinstance(val, bool) or isinstance(val, int):
        probs_dict = {}
        samples = [sample[fi] for sample in samples]
        for sample in set(samples):
            row[fi] = sample
            probs_dict[sample] = numpy.exp(
                server.score(row) - base_score)
        if len(probs_dict) == 1:
            assert_almost_equal(probs_dict[sample], 1., places=SCORE_TOLERANCE)
            return
        if min(probs_dict.values()) < MIN_CATEGORICAL_PROB:
            return
        gof = discrete_goodness_of_fit(samples, probs_dict, plot=True)
    elif isinstance(val, float):
        probs = numpy.exp([
            server.score(sample) - base_score
            for sample in samples
        ])
        samples = [sample[fi] for sample in samples]
        gof = density_goodness_of_fit(samples, probs, plot=True)
    assert_greater(gof, MIN_GOODNESS_OF_FIT)


@for_each_dataset
def test_samples_match_scores(root, rows, **unused):
    rows = load_rows(rows)
    rows = rows[::len(rows) / 5]
    with loom.query.get_server(root, debug=True) as server:
        for row in rows:
            _check_marginal_samples_match_scores(server, row, 0)
