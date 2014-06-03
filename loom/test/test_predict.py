import os
from itertools import izip
from nose.tools import assert_false, assert_equal
from distributions.fileutil import tempdir
from loom.test.util import for_each_dataset
from loom.test.util import CLEANUP_ON_ERROR
import loom.predict

CONFIG = {}


@for_each_dataset
def test_predict(model, groups, **unused):
    queries = loom.predict.get_example_queries(model)
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


@for_each_dataset
def test_server(model, groups, **unused):
    queries = loom.predict.get_example_queries(model)
    with tempdir(cleanup_on_error=CLEANUP_ON_ERROR):
        config_in = os.path.abspath('config.pb.gz')
        loom.config.config_dump(CONFIG, config_in)
        kwargs = {
            'config_in': config_in,
            'model_in': model,
            'groups_in': groups,
            'debug': True,
        }
        with loom.predict.Server(**kwargs) as predict:
            results = []
            for i, query in enumerate(queries):
                print 'prediction {}'.format(i)
                results.append(predict(query))
            print 'done'
    for query, result in izip(queries, results):
        assert_equal(query.id, result.id)
        assert_false(hasattr(query, 'error'))
        assert_equal(len(result.samples), 1)
