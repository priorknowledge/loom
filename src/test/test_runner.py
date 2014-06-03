import os
from itertools import izip
from nose.tools import assert_true, assert_false, assert_equal
from loom.test.util import for_each_dataset
from distributions.fileutil import tempdir
from distributions.io.stream import open_compressed, protobuf_stream_load
from loom.schema_pb2 import ProductModel, CrossCat
from loom.test.util import CLEANUP_ON_ERROR
import loom.config
import loom.runner
import loom.predict

CONFIGS = [
    {
        'schedule': {'extra_passes': 0.0},
    },
    {
        'schedule': {'extra_passes': 1.5},
        'kernels': {
            'cat': {
                'empty_group_count': 1,
                'row_queue_capacity': 0,
            },
            'kind': {'iterations': 0},
        },
    },
    {
        'schedule': {'extra_passes': 1.5},
        'kernels': {
            'cat': {
                'empty_group_count': 1,
                'row_queue_capacity': 8,
            },
            'kind': {'iterations': 0},
        },
    },
    {
        'schedule': {'extra_passes': 1.5},
        'kernels': {
            'cat': {
                'empty_group_count': 1,
                'row_queue_capacity': 0,
            },
            'kind': {
                'iterations': 1,
                'empty_kind_count': 1,
                'row_queue_capacity': 0,
                'score_parallel': False,
            },
        },
    },
    {
        'schedule': {'extra_passes': 1.5, 'max_reject_iters': 1},
        'kernels': {
            'cat': {
                'empty_group_count': 1,
                'row_queue_capacity': 0,
            },
            'kind': {
                'iterations': 1,
                'empty_kind_count': 1,
                'row_queue_capacity': 0,
                'score_parallel': False,
            },
        },
    },
    {
        'schedule': {'extra_passes': 1.5, 'max_reject_iters': 1},
        'kernels': {
            'cat': {
                'empty_group_count': 1,
                'row_queue_capacity': 0,
            },
            'kind': {
                'iterations': 1,
                'empty_kind_count': 1,
                'row_queue_capacity': 0,
                'score_parallel': False,
            },
        },
    },
    {
        'schedule': {'extra_passes': 1.5, 'max_reject_iters': 100},
        'kernels': {
            'cat': {
                'empty_group_count': 1,
                'row_queue_capacity': 0,
            },
            'kind': {
                'iterations': 1,
                'empty_kind_count': 1,
                'row_queue_capacity': 0,
                'score_parallel': False,
            },
        },
    },
    {
        'schedule': {'extra_passes': 1.5, 'max_reject_iters': 100},
        'kernels': {
            'cat': {
                'empty_group_count': 1,
                'row_queue_capacity': 0,
            },
            'kind': {
                'iterations': 1,
                'empty_kind_count': 1,
                'row_queue_capacity': 8,
                'score_parallel': True,
            },
        },
    },
]


def get_group_counts(groups_out):
    group_counts = []
    for f in os.listdir(groups_out):
        group_count = 0
        groups = os.path.join(groups_out, f)
        for string in protobuf_stream_load(groups):
            group = ProductModel.Group()
            group.ParseFromString(string)
            group_count += 1
        group_counts.append(group_count)
    assert group_counts, 'no groups found'
    return group_counts


@for_each_dataset
def test_shuffle(rows, **unused):
    with tempdir(cleanup_on_error=CLEANUP_ON_ERROR):
        seed = 12345
        rows_out = os.path.abspath('rows_out.pbs.gz')
        loom.runner.shuffle(
            rows_in=rows,
            rows_out=rows_out,
            seed=seed)
        assert_true(os.path.exists(rows_out))


@for_each_dataset
def test_infer(rows, model, **unused):
    with tempdir(cleanup_on_error=CLEANUP_ON_ERROR):
        row_count = sum(1 for _ in protobuf_stream_load(rows))
        with open_compressed(model) as f:
            message = CrossCat()
            message.ParseFromString(f.read())
        kind_count = len(message.kinds)

        for config in CONFIGS:
            loom.config.fill_in_defaults(config)
            schedule = config['schedule']
            print 'config: {}'.format(config)

            greedy = (schedule['extra_passes'] == 0)
            kind_iters = config['kernels']['kind']['iterations']
            fixed_kind_structure = greedy or kind_iters == 0

            with tempdir(cleanup_on_error=CLEANUP_ON_ERROR):
                config_in = os.path.abspath('config.pb.gz')
                model_out = os.path.abspath('model.pb.gz')
                groups_out = os.path.abspath('groups')
                assign_out = os.path.abspath('assign.pbs.gz')
                log_out = os.path.abspath('log.pbs.gz')
                os.mkdir(groups_out)
                loom.config.config_dump(config, config_in)
                loom.runner.infer(
                    config_in=config_in,
                    rows_in=rows,
                    model_in=model,
                    model_out=model_out,
                    groups_out=groups_out,
                    assign_out=assign_out,
                    log_out=log_out,
                    debug=True,)

                if fixed_kind_structure:
                    assert_equal(len(os.listdir(groups_out)), kind_count)

                group_counts = get_group_counts(groups_out)

                assign_count = sum(1 for _ in protobuf_stream_load(assign_out))
                assert_equal(assign_count, row_count)

            print 'row_count: {}'.format(row_count)
            print 'group_counts: {}'.format(' '.join(map(str, group_counts)))
            for group_count in group_counts:
                assert_true(
                    group_count <= row_count,
                    'groups are all singletons')


@for_each_dataset
def test_posterior_enum(rows, model, **unused):
    with tempdir(cleanup_on_error=CLEANUP_ON_ERROR):
        config_in = os.path.abspath('config.pb.gz')
        config = {
            'posterior_enum': {
                'sample_count': 7,
            },
            'kernels': {
                'kind': {
                    'row_queue_capacity': 0,
                    'score_parallel': False,
                },
            },
        }
        loom.config.config_dump(config, config_in)
        assert_true(os.path.exists(config_in))

        samples_out = os.path.abspath('samples.pbs.gz')
        loom.runner.posterior_enum(
            config_in=config_in,
            model_in=model,
            rows_in=rows,
            samples_out=samples_out,
            debug=True)
        assert_true(os.path.exists(samples_out))
        actual_count = sum(1 for _ in protobuf_stream_load(samples_out))
        assert_equal(actual_count, config['posterior_enum']['sample_count'])


@for_each_dataset
def test_generate(model, **unused):
    for row_count in [0, 1, 100]:
        for density in [0.0, 0.5, 1.0]:
            with tempdir(cleanup_on_error=CLEANUP_ON_ERROR):
                config_in = os.path.abspath('config.pb.gz')
                config = {
                    'generate': {
                        'row_count': row_count,
                        'density': density,
                    },
                }
                loom.config.config_dump(config, config_in)
                assert_true(os.path.exists(config_in))

                rows_out = os.path.abspath('rows.pbs.gz')
                model_out = os.path.abspath('model.pb.gz')
                groups_out = os.path.abspath('groups')
                loom.runner.generate(
                    config_in=config_in,
                    model_in=model,
                    rows_out=rows_out,
                    model_out=model_out,
                    groups_out=groups_out,
                    debug=True)
                assert_true(os.path.exists(rows_out))
                assert_true(os.path.exists(model_out))
                assert_true(os.path.exists(groups_out))

                group_counts = get_group_counts(groups_out)
                print 'group_counts: {}'.format(
                    ' '.join(map(str, group_counts)))


@for_each_dataset
def test_predict(model, groups, **unused):
    with tempdir(cleanup_on_error=CLEANUP_ON_ERROR):
        config_in = os.path.abspath('config.pb.gz')
        loom.config.config_dump({}, config_in)
        queries = loom.predict.get_example_queries(model)
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
