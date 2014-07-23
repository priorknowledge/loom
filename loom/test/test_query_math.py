import loom.preql
import loom.query
from loom.query import (
    SingleSampleProtobufServer,
    MultiSampleProtobufServer
)
from distributions.fileutil import tempdir
from distributions.util import (
    density_goodness_of_fit,
    discrete_goodness_of_fit,
)
from nose.tools import (
    assert_greater,
    assert_almost_equal,
    assert_less_equal,
)
from nose import SkipTest
import numpy as np
import scipy.stats
from loom.test.util import for_each_dataset, CLEANUP_ON_ERROR
from itertools import product
from test_query import get_protobuf_server
from util import load_rows


GOF_EXP = 3
MIN_GOODNESS_OF_FIT = 10. ** (-GOF_EXP)
CONFIDENCE_INTERVAL = 1. - MIN_GOODNESS_OF_FIT
Z_SCORE = scipy.stats.norm.ppf(CONFIDENCE_INTERVAL)

SAMPLE_COUNT = 300

# tests are inaccurate with highly imbalanced data
MIN_CATEGORICAL_PROB = .01


@for_each_dataset
def test_score_none(model, groups, encoding, **unused):
    argss = [
        (SingleSampleProtobufServer, model, groups),
        (MultiSampleProtobufServer, [model], [groups]),
        (MultiSampleProtobufServer, [model, model], [groups, groups])
    ]
    with tempdir(cleanup_on_error=CLEANUP_ON_ERROR):
        for args in argss:
            protobuf_server = get_protobuf_server(*args)
            query_server = loom.query.QueryServer(protobuf_server)
            preql = loom.preql.PreQL(query_server, encoding)
            fnames = preql.feature_names
            assert_almost_equal(
                query_server.score([None for _ in fnames]),
                0.,
                places=GOF_EXP)


@for_each_dataset
def test_mi_entropy_relations(model, groups, encoding, **unused):
    with tempdir(cleanup_on_error=CLEANUP_ON_ERROR):
        protobuf_server = get_protobuf_server(
            loom.query.SingleSampleProtobufServer,
            model,
            groups)
        query_server = loom.query.QueryServer(protobuf_server)
        preql = loom.preql.PreQL(query_server, encoding)
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
            mutual_info = query_server.mutual_information(
                to_sample1,
                to_sample2,
                sample_count=SAMPLE_COUNT)
            entropy1 = query_server.entropy(
                to_sample1,
                sample_count=SAMPLE_COUNT)
            entropy2 = query_server.entropy(
                to_sample2,
                sample_count=SAMPLE_COUNT)
            entropy_joint = query_server.entropy(
                to_sample,
                sample_count=SAMPLE_COUNT)
            if to_sample1 == to_sample2:
                measures = [mutual_info, entropy1, entropy2, entropy_joint]
                for m1, m2 in product(measures, measures):
                    assert_less_equal(
                        abs(m1.mean - m2.mean),
                        Z_SCORE * np.sqrt(m1.variance + m2.variance))
            expected = mutual_info.mean
            actual = entropy1.mean + entropy2.mean - entropy_joint.mean
            variance = sum(term.variance for term in [
                mutual_info, entropy1, entropy2, entropy_joint])
            error = np.sqrt(variance)
            assert_less_equal(abs(actual - expected), Z_SCORE * error)


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
            probs_dict[sample] = np.exp(query_server.score(row) - base_score)
        if len(probs_dict) == 1:
            assert_almost_equal(probs_dict[sample], 1., places=GOF_EXP)
            return
        if min(probs_dict.values()) < MIN_CATEGORICAL_PROB:
            return
        gof = discrete_goodness_of_fit(samples, probs_dict, plot=True)
    elif isinstance(val, float):
        probs = np.exp([query_server.score(sample) - base_score
                       for sample in samples])
        samples = [sample[fi] for sample in samples]
        gof = density_goodness_of_fit(samples, probs, plot=True)
    assert_greater(gof, MIN_GOODNESS_OF_FIT)


@for_each_dataset
def test_samples_match_scores_one(model, groups, rows, **unused):
    argss = [
        (SingleSampleProtobufServer, model, groups),
    ]
    rows = load_rows(rows)
    rows = rows[::len(rows) / 2]
    with tempdir(cleanup_on_error=CLEANUP_ON_ERROR):
        for row in rows:
            for args in argss:
                with get_protobuf_server(*args) as protobuf_server:
                    _check_marginal_samples_match_scores(
                        protobuf_server,
                        row,
                        0)


@for_each_dataset
def test_samples_match_scores_multi(model, groups, rows, **unused):
    argss = [
        (MultiSampleProtobufServer, [model], [groups]),
        (MultiSampleProtobufServer, [model, model], [groups, groups]),
    ]
    rows = load_rows(rows)
    rows = rows[::len(rows) / 2]
    with tempdir(cleanup_on_error=CLEANUP_ON_ERROR):
        for row in rows:
            for args in argss:
                raise SkipTest(
                    "TODO: differenct seeds for multi-sample servers")
