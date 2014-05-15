import os
from nose.tools import assert_true
from loom.test.util import for_each_dataset
from distributions.io.stream import json_load, json_dump
from distributions.fileutil import tempdir
import loom.tardis

CLEANUP_ON_ERROR = int(os.environ.get('CLEANUP_ON_ERROR', 1))


@for_each_dataset
def test_run(meta, data, mask, tardis_conf, latent, **unused):
    with tempdir(cleanup_on_error=CLEANUP_ON_ERROR):
        sample_out = os.path.abspath('sample_000.json')
        scores_out = os.path.abspath('scores_000.json')
        log_config = {'tags': {
            'seed': 0,
            'experiment': 'loom.test.test_tardis',
        }}

        cheaper_conf = os.path.abspath('tardis_conf.json')
        conf = json_load(tardis_conf)
        conf['schedule'] = {
            'kind_passes': 1.0,
            'cat_passes': 1.0,
        }
        json_dump(conf, cheaper_conf)

        loom.tardis.run(
            cheaper_conf,
            meta,
            data,
            mask,
            latent,
            sample_out,
            scores_out,
            log_config)

        assert_true(os.path.exists(sample_out))
        assert_true(os.path.exists(scores_out))
