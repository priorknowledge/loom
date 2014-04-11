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
        assign = os.path.abspath('assign.pbs.gz')
        print meta, latent, model, groups, assign
        loom.format.import_latent(meta, latent, model, groups, assign)
        assert_true(os.path.exists(os.path.join(groups, 'mixture.000.pbs.gz')))
        assert_true(os.path.exists(model))
        assert_true(os.path.exists(assign))


@for_each_dataset
def test_export_latent(meta, latent, **unused):
    with tempdir():
        model = os.path.abspath('model.pb.gz')
        groups = 'groups'
        loom.format.import_latent(meta, latent, model, groups)
        assert_true(os.path.exists(model))
        latent_out = os.path.abspath('latent.json')
        groups = None  # TODO export groups
        assign = None  # TODO export assign
        loom.format.export_latent(meta, model, latent_out, groups, assign)
        assert_true(os.path.exists(latent_out))
        json_load(latent_out)
