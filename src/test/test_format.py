import os
from nose.tools import assert_true
from loom.test.util import for_each_dataset
from distributions.fileutil import tempdir
from distributions.io.stream import json_load
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
        print meta, latent, model, groups
        loom.format.import_latent(meta, latent, model, groups)
        assert_true(
            groups is None or
            os.path.exists(os.path.join(groups, 'mixture.000.pbs.gz')))
        assert_true(os.path.exists(model))


@for_each_dataset
def test_export_latent(meta, latent, **unused):
    with tempdir():
        model = os.path.abspath('model.pb.gz')
        groups = 'groups'
        loom.format.import_latent(meta, latent, model, groups)
        assert_true(os.path.exists(model))
        latent_out = os.path.abspath('latent.json')
        groups = None  # TODO export groups
        loom.format.export_latent(meta, model, latent_out, groups)
        assert_true(os.path.exists(latent_out))
        json_load(latent_out)
