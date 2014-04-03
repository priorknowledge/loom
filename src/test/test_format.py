import os
from nose.tools import assert_true
from loom.test.util import for_each_dataset
from distributions.fileutil import tempdir
import loom.format


@for_each_dataset
def test_import_data(meta, data, mask, **kwargs):
    with tempdir():
        model = os.path.abspath('model.pb2')
        print meta, data, mask, model
        loom.format.import_data(meta, data, mask, model)
        assert_true(os.path.exists(model))
