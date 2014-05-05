import os
from nose.tools import assert_true, assert_equal
from loom.test.util import for_each_dataset
from distributions.fileutil import tempdir
from distributions.io.stream import json_load, protobuf_stream_load
from loom.schema_pb2 import ProductModel
import loom.format
import loom.runner

CLEANUP_ON_ERROR = int(os.environ.get('CLEANUP_ON_ERROR', 1))

CONFIGS = [
    {'extra_passes': 0.0, 'kind_count': 0, 'kind_iters': 0, 'groups': True},
    {'extra_passes': 1.5, 'kind_count': 0, 'kind_iters': 0, 'groups': True},
    {'extra_passes': 1.5, 'kind_count': 0, 'kind_iters': 0, 'groups': False},
    {'extra_passes': 1.5, 'kind_count': 1, 'kind_iters': 1, 'groups': False},
    {'extra_passes': 1.5, 'kind_count': 4, 'kind_iters': 4, 'groups': False},
]


@for_each_dataset
def test_infer(meta, data, mask, latent, predictor, **unused):
    with tempdir(cleanup_on_error=CLEANUP_ON_ERROR):
        model = os.path.abspath('model.pb.gz')
        groups_in = os.path.abspath('groups')
        loom.format.import_latent(meta, latent, model, groups_in)
        assert_true(os.path.exists(model))
        assert_true(groups_in is None or os.path.exists(groups_in))
        row_count = len(json_load(meta)['object_pos'])
        kind_count = len(json_load(predictor)['structure'])

        rows = os.path.abspath('rows.pbs.gz')
        loom.format.import_data(meta, data, mask, rows)
        assert_true(os.path.exists(rows))

        for config in CONFIGS:
            print 'config: {}'.format(config)
            config = config.copy()
            if config.pop('groups'):
                config['groups_in'] = groups_in
            fixed_kind_structure = (config['kind_count'] == 0)

            with tempdir(cleanup_on_error=CLEANUP_ON_ERROR):
                model_out = os.path.abspath('model.pb.gz')
                groups_out = os.path.abspath('groups')
                assign_out = os.path.abspath('assign.pbs.gz')
                os.mkdir(groups_out)
                loom.runner.infer(
                    model_in=model,
                    rows_in=rows,
                    model_out=model_out,
                    groups_out=groups_out,
                    assign_out=assign_out,
                    debug=True,
                    **config)

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

            print 'row_count: {}'.format(row_count)
            print 'group_counts: {}'.format(' '.join(map(str, group_counts)))
            for group_count in group_counts:
                assert_true(
                    group_count <= row_count,
                    'groups are all singletons')


@for_each_dataset
def test_posterior_enum(meta, data, mask, latent, **unused):
    with tempdir():
        model = os.path.abspath('model.pb.gz')
        loom.format.import_latent(meta, latent, model)
        assert_true(os.path.exists(model))

        rows = os.path.abspath('rows.pbs.gz')
        loom.format.import_data(meta, data, mask, rows)
        assert_true(os.path.exists(rows))

        samples_out = os.path.abspath('samples.pbs.gz')
        sample_count = 7
        loom.runner.posterior_enum(
            model_in=model,
            rows_in=rows,
            samples_out=samples_out,
            sample_count=sample_count,
            debug=True)
        assert_true(os.path.exists(samples_out))
        actual_count = sum(1 for _ in protobuf_stream_load(samples_out))
        assert_equal(actual_count, sample_count)
