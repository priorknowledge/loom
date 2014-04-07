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
        loom.format.import_data(meta, data, mask, values)
        assert_true(os.path.exists(values))


@for_each_dataset
def test_import_latent(meta, latent, **unused):
    with tempdir():
        model = os.path.abspath('model.pb.gz')
        print meta, latent, model
        loom.format.import_latent(meta, latent, model)
        assert_true(os.path.exists(model))


@for_each_dataset
def test_export_latent(meta, latent, **unused):
    with tempdir():
        model = os.path.abspath('model.pb.gz')
        loom.format.import_latent(meta, latent, model)
        assert_true(os.path.exists(model))
        latent_out = os.path.abspath('latent.json')
        loom.format.export_latent(meta, model, latent_out)
        assert_true(os.path.exists(latent_out))
        json_load(latent_out)
