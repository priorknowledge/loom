import os
from nose.tools import assert_true
from loom.test.util import for_each_dataset
from distributions.fileutil import tempdir
import loom.format
import loom.runner


@for_each_dataset
def test_loom(meta, data, mask, latent, **unused):
    with tempdir():
        model = os.path.abspath('model.pb2')
        loom.format.import_latent(meta, latent, model)
        assert_true(os.path.exists(model))

        values = os.path.abspath('values.pb2stream')
        loom.format.import_data(meta, data, mask, values)
        assert_true(os.path.exists(values))

        groups = os.path.abspath('groups.pb2')
        loom.runner.run(model, values, groups)
        assert_true(os.path.exists(groups))
