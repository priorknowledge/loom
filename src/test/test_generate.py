import os
from distributions.fileutil import tempdir
import loom.generate

CLEANUP_ON_ERROR = int(os.environ.get('CLEANUP_ON_ERROR', 1))

FEATURE_TYPES = loom.schema.FEATURE_TYPES.keys()
FEATURE_TYPES += [None]


def test_generate():
    for feature_type in FEATURE_TYPES:
        yield _test_generate, feature_type


def _test_generate(feature_type):
    root = os.path.abspath(os.path.curdir)
    with tempdir(cleanup_on_error=CLEANUP_ON_ERROR):
        model_out = os.path.abspath('model.pb.gz')
        rows_out = os.path.abspath('rows.pbs.gz')
        os.chdir(root)
        loom.generate.generate(
            row_count=100,
            feature_count=100,
            feature_type=feature_type,
            density=0.5,
            model_out=model_out,
            rows_out=rows_out,
            debug=True,
            profile=None)
