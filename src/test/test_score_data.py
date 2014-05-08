import os
from nose.tools import assert_true
from loom.test.util import for_each_dataset
from distributions.fileutil import tempdir
from distributions.io.stream import json_load, protobuf_stream_load
from distributions.tests.util import assert_close
from loom.schema_pb2 import PosteriorEnum
import loom.format
import loom.runner

CONFIGS = [
    {'kind_count': 0, 'kind_iters': 0},
    {'kind_count': 10, 'kind_iters': 10},
]


@for_each_dataset
def test_score_data(meta, data, mask, latent, scores, **unused):
    with tempdir():
        model = os.path.abspath('model.pb.gz')
        groups = os.path.abspath('groups')
        assign = os.path.abspath('assign.pbs.gz')
        rows = os.path.abspath('rows.pbs.gz')

        loom.format.import_latent(
            meta_in=meta,
            latent_in=latent,
            model_out=model,
            groups_out=groups,
            assign_out=assign)
        loom.format.import_data(
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
            print 'config: {}'.format(config)

            with tempdir():
                samples = os.path.abspath('samples.pbs.gz')
                loom.runner.posterior_enum(
                    model_in=model,
                    rows_in=rows,
                    groups_in=groups,
                    assign_in=assign,
                    samples_out=samples,
                    sample_count=1,
                    sample_skip=0,
                    **config)

                message = PosteriorEnum.Sample()
                for string in protobuf_stream_load(samples):
                    message.ParseFromString(string)
                    actual_score = message.score

                assert_close(expected_score, actual_score)
