import os
from nose.tools import assert_true
from loom.compat.test.util import for_each_dataset
from distributions.fileutil import tempdir
from distributions.io.stream import json_load, protobuf_stream_load
from distributions.tests.util import assert_close
from loom.schema_pb2 import PosteriorEnum
import loom.config
import loom.compat.format
import loom.runner

CONFIGS = [
    {
        'posterior_enum': {'sample_count': 1, 'sample_skip': 0},
        'kernels': {'kind': {'iterations': 0}},
    },
    {
        'posterior_enum': {'sample_count': 1, 'sample_skip': 0},
        'kernels': {'kind': {'iterations': 10}},
    },
]


@for_each_dataset
def test_score_data(meta, data, mask, latent, scores, **unused):
    with tempdir():
        model = os.path.abspath('model.pb.gz')
        groups = os.path.abspath('groups')
        assign = os.path.abspath('assign.pbs.gz')
        rows = os.path.abspath('rows.pbs.gz')

        loom.compat.format.import_latent(
            meta_in=meta,
            latent_in=latent,
            model_out=model,
            groups_out=groups,
            assign_out=assign)
        loom.compat.format.import_data(
            meta_in=meta,
            data_in=data,
            mask_in=mask,
            rows_out=rows)

        assert_true(os.path.exists(model))
        assert_true(os.path.exists(groups))
        assert_true(os.path.exists(assign))
        assert_true(os.path.exists(rows))

        expected_score = json_load(scores)['score']

        for config in CONFIGS:
            with tempdir():
                print 'config: {}'.format(config)
                config_in = os.path.abspath('config.pb.gz')
                loom.config.config_dump(config, config_in)
                assert_true(os.path.exists(config_in))

                samples = os.path.abspath('samples.pbs.gz')
                loom.runner.posterior_enum(
                    config_in=config_in,
                    model_in=model,
                    rows_in=rows,
                    groups_in=groups,
                    assign_in=assign,
                    samples_out=samples)

                message = PosteriorEnum.Sample()
                for string in protobuf_stream_load(samples):
                    message.ParseFromString(string)
                    actual_score = message.score

                assert_close(expected_score, actual_score)
