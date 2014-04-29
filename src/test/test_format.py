import os
from nose.tools import assert_true
from loom.test.util import for_each_dataset
from distributions.fileutil import tempdir
from distributions.io.stream import json_load
from distributions.tests.util import assert_close
import loom.format


@for_each_dataset
def test_import_data(meta, data, mask, **unused):
    with tempdir():
        values = os.path.abspath('values.pbs.gz')
        print meta, data, mask, values
        loom.format.import_data(meta, data, mask, values, validate=True)
        assert_true(os.path.exists(values))


@for_each_dataset
def test_import_latent(meta, latent, **unused):
    with tempdir():
        model = os.path.abspath('model.pb.gz')
        groups = os.path.abspath('groups')
        assign = os.path.abspath('assign.pbs.gz')
        print meta, latent, model, groups, assign
        loom.format.import_latent(meta, latent, model, groups, assign)
        assert_true(os.path.exists(os.path.join(groups, 'mixture.000.pbs.gz')))
        assert_true(os.path.exists(model))
        assert_true(os.path.exists(assign))


@for_each_dataset
def test_export_latent(meta, latent, **unused):
    with tempdir(cleanup_on_error=False):
        model = os.path.abspath('model.pb.gz')
        groups = os.path.abspath('groups')
        assign = os.path.abspath('assign.pbs.gz')
        loom.format.import_latent(meta, latent, model, groups, assign)
        assert_true(os.path.exists(model))
        latent_out = os.path.abspath('latent.json')
        loom.format.export_latent(meta, model, latent_out, groups, assign)
        assert_true(os.path.exists(latent_out))
        actual = json_load(latent_out)
        expected = json_load(latent)
        expected.pop('meta', None)  # ignore latent['meta']
        meta = json_load(meta)
        loom.format.canonicalize_latent(meta, actual)
        loom.format.canonicalize_latent(meta, expected)
        assert_close(actual, expected)
