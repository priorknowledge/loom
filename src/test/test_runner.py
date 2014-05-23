import os
from nose.tools import assert_true, assert_equal
from loom.test.util import for_each_dataset
from distributions.fileutil import tempdir
from distributions.io.stream import json_load, protobuf_stream_load
from loom.schema_pb2 import ProductModel
import loom.config
import loom.format
import loom.runner

CLEANUP_ON_ERROR = int(os.environ.get('CLEANUP_ON_ERROR', 1))

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
    # FIXME
    #{
    #    'schedule': {'extra_passes': 1.5},
    #    'kernels': {
    #        'cat': {
    #            'empty_group_count': 1,
    #            'row_queue_capacity': 8,
    #        },
    #        'kind': {'iterations': 0},
    #    },
    #},
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


@for_each_dataset
def test_shuffle(meta, data, mask, **unused):
    with tempdir(cleanup_on_error=CLEANUP_ON_ERROR):
        rows_in = os.path.abspath('rows_in.pbs.gz')
        loom.format.import_data(
            meta_in=meta,
            data_in=data,
            mask_in=mask,
            rows_out=rows_in)
        assert_true(os.path.exists(rows_in))

        seed = 12345
        rows_out = os.path.abspath('rows_out.pbs.gz')
        loom.runner.shuffle(
            rows_in=rows_in,
            rows_out=rows_out,
            seed=seed)
        assert_true(os.path.exists(rows_out))


@for_each_dataset
def test_infer(meta, data, mask, tardis_conf, latent, predictor, **unused):
    with tempdir(cleanup_on_error=CLEANUP_ON_ERROR):
        model = os.path.abspath('model.pb.gz')
        groups_in = os.path.abspath('groups')
        loom.format.import_latent(
            meta_in=meta,
            latent_in=latent,
            tardis_conf_in=tardis_conf,
            model_out=model,
            groups_out=groups_in)
        assert_true(os.path.exists(model))
        assert_true(groups_in is None or os.path.exists(groups_in))
        row_count = len(json_load(meta)['object_pos'])
        kind_count = len(json_load(predictor)['structure'])

        rows = os.path.abspath('rows.pbs.gz')
        loom.format.import_data(
            meta_in=meta,
            data_in=data,
            mask_in=mask,
            rows_out=rows)
        assert_true(os.path.exists(rows))

        for config in CONFIGS:
            loom.config.fill_in_defaults(config)
            schedule = config['schedule']
            print 'config: {}'.format(config)

            greedy = (schedule['extra_passes'] == 0)
            if greedy:
                groups = groups_in
            else:
                groups = None

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
                    groups_in=groups,
                    model_out=model_out,
                    groups_out=groups_out,
                    assign_out=assign_out,
                    log_out=log_out,
                    debug=True,)

                if fixed_kind_structure:
                    assert_equal(len(os.listdir(groups_out)), kind_count)

                group_counts = []
                for f in os.listdir(groups_out):
                    group_count = 0
                    groups = os.path.join(groups_out, f)
                    for string in protobuf_stream_load(groups):
                        group = ProductModel.Group()
                        group.ParseFromString(string)
                        group_count += 1
                    group_counts.append(group_count)

                assign_count = sum(1 for _ in protobuf_stream_load(assign_out))
                assert_equal(assign_count, row_count)

            print 'row_count: {}'.format(row_count)
            print 'group_counts: {}'.format(' '.join(map(str, group_counts)))
            for group_count in group_counts:
                assert_true(
                    group_count <= row_count,
                    'groups are all singletons')


@for_each_dataset
def test_posterior_enum(meta, data, mask, latent, **unused):
    with tempdir(cleanup_on_error=CLEANUP_ON_ERROR):
        model = os.path.abspath('model.pb.gz')
        loom.format.import_latent(
            meta_in=meta,
            latent_in=latent,
            model_out=model)
        assert_true(os.path.exists(model))

        rows = os.path.abspath('rows.pbs.gz')
        loom.format.import_data(
            meta_in=meta,
            data_in=data,
            mask_in=mask,
            rows_out=rows)
        assert_true(os.path.exists(rows))

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
