import os
from distributions.fileutil import tempdir
from loom.test.util import CLEANUP_ON_ERROR
import loom.generate

FEATURE_TYPES = loom.schema.FEATURE_TYPES.keys()
FEATURE_TYPES += ['mixed']


def test_generate():
    for feature_type in FEATURE_TYPES:
        yield _test_generate, feature_type


def _test_generate(feature_type):
    root = os.path.abspath(os.path.curdir)
    with tempdir(cleanup_on_error=CLEANUP_ON_ERROR):
        init_out = os.path.abspath('init.pb.gz')
        rows_out = os.path.abspath('rows.pbs.gz')
        model_out = os.path.abspath('model.pb.gz')
        groups_out = os.path.abspath('groups')
        os.chdir(root)
        loom.generate.generate(
            feature_type=feature_type,
            row_count=100,
            feature_count=100,
            density=0.5,
            init_out=init_out,
            rows_out=rows_out,
            model_out=model_out,
            groups_out=groups_out,
            debug=True,
            profile=None)
