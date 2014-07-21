import loom.preql
import loom.query
from loom.query import (
    SingleSampleProtobufServer,
    MultiSampleProtobufServer
)
from distributions.fileutil import tempdir
from distributions.util import density_goodness_of_fit, discrete_goodness_of_fit
from nose.tools import (
    assert_greater,
    assert_almost_equal,
    assert_less_equal,
)
# from nose import SkipTest
import numpy as np
from loom.test.util import for_each_dataset, CLEANUP_ON_ERROR
from itertools import product
from test_query import get_protobuf_server
from util import load_rows


MIN_GOODNESS_OF_FIT = 1e-3

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
                places=3)



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
            [fnames[0], fnames[1]],
            [fnames[2], fnames[3]],
        ]
        for fset1, fset2 in product(feature_sets, feature_sets):
            to_sample1 = preql.cols_to_sample(fset1)
            to_sample2 = preql.cols_to_sample(fset2)
            to_sample = preql.cols_to_sample(fset1 + fset2)
            mutual_info = query_server.mutual_information(
                to_sample1,
                to_sample2,
                sample_count=100)
            entropy1 = query_server.entropy(
                to_sample1,
                sample_count=100)
            entropy2 = query_server.entropy(
                to_sample2,
                sample_count=100)
            entropy_joint = query_server.entropy(
                to_sample,
                sample_count=100)
            if to_sample1 == to_sample2:
                measures = [mutual_info, entropy1, entropy2, entropy_joint]
                for m1, m2 in product(measures, measures):
                    assert_less_equal(
                        abs(m1.mean - m2.mean),
                        2.25 * (m1.variance + m2.variance))
            expected = mutual_info.mean
            actual = entropy1.mean + entropy2.mean - entropy_joint.mean
            err = sum(term.variance for term in [
                mutual_info, entropy1, entropy2, entropy_joint])
            assert_less_equal(abs(actual - expected), 2.25 * err)

def _check_marginal_samples_match_scores(protobuf_server, row, fi):
    query_server = loom.query.QueryServer(protobuf_server)
    cond_row = loom.query.protobuf_to_data_row(row.data)
    cond_row[fi] = None
    to_sample = [i == fi for i in range(len(cond_row))]
    samples = query_server.sample(to_sample, cond_row, 500)
    val = samples[0][fi]
    base_score = query_server.score(cond_row)
    if isinstance(val, bool) or isinstance(val, int):
        probs_dict = {}
        samples = [sample[fi] for sample in samples]
        for sample in set(samples):
            row = cond_row
            row[fi] = sample
            probs_dict[sample] = np.exp(query_server.score(row) - base_score)
        if len(probs_dict) == 1:
            assert_almost_equal(probs_dict[sample], 1., places=2)
            return
        if min(probs_dict.values()) < .01:
            # TODO tests are inaccurate with highly imbalanced data
            return
        gof = discrete_goodness_of_fit(samples, probs_dict, plot=True)
    elif isinstance(val, float):
        probs = np.exp([query_server.score(sample) - base_score for sample in samples])
        samples = [sample[fi] for sample in samples]
        gof = density_goodness_of_fit(samples, probs, plot=True)
    assert_greater(gof, MIN_GOODNESS_OF_FIT)


@for_each_dataset
def test_samples_match_scores(model, groups, rows, **unused):
    argss = [
        (SingleSampleProtobufServer, model, groups),
        (MultiSampleProtobufServer, [model], [groups]),
        (MultiSampleProtobufServer, [model, model], [groups, groups])
    ]
    rows = load_rows(rows)
    rows = rows[::len(rows)/2]
    with tempdir(cleanup_on_error=CLEANUP_ON_ERROR):
        for row in rows:
            for args in argss:
                with get_protobuf_server(*args) as protobuf_server:
                    if len(args[1]) > 1:
                        # TODO: different seeds for multi sample servers
                        continue
                    _check_marginal_samples_match_scores(protobuf_server, row, 0)
