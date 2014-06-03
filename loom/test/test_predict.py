import os
from itertools import izip
from nose.tools import assert_false, assert_equal
from distributions.fileutil import tempdir
from distributions.io.stream import open_compressed
from loom.schema_pb2 import CrossCat, PreQL
from loom.test.util import for_each_dataset
from loom.test.util import CLEANUP_ON_ERROR
import loom.predict

CONFIG = {}


def get_example_queries(model):
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
        observed = none_observed[:]
        observed[f] = True
        observeds.append(observed)
    observeds.append(none_observed)

    queries = []
    for i, observed in enumerate(observeds):
        query = PreQL.Predict.Query()
        query.id = "example-{}".format(i)
        query.data.observed[:] = none_observed
        query.to_predict[:] = observed
        query.sample_count = 1
        queries.append(query)

    return queries


@for_each_dataset
def test_server(model, groups, **unused):
    queries = get_example_queries(model)
    with tempdir(cleanup_on_error=CLEANUP_ON_ERROR):
        config_in = os.path.abspath('config.pb.gz')
        loom.config.config_dump(CONFIG, config_in)
        kwargs = {
            'config_in': config_in,
            'model_in': model,
            'groups_in': groups,
            'debug': True,
        }
        with loom.predict.serve(**kwargs) as predict:
            results = [predict(query) for query in queries]
    for query, result in izip(queries, results):
        assert_equal(query.id, result.id)
        assert_false(hasattr(query, 'error'))
        assert_equal(len(result.samples), 1)


@for_each_dataset
def test_batch_predict(model, groups, **unused):
    queries = get_example_queries(model)
    with tempdir(cleanup_on_error=CLEANUP_ON_ERROR):
        config_in = os.path.abspath('config.pb.gz')
        loom.config.config_dump(CONFIG, config_in)
        results = loom.predict.batch_predict(
            config_in=config_in,
            model_in=model,
            groups_in=groups,
            queries=queries,
            debug=True)
    assert_equal(len(results), len(queries))
    for query, result in izip(queries, results):
        assert_equal(query.id, result.id)
        assert_false(hasattr(query, 'error'))
        assert_equal(len(result.samples), 1)
