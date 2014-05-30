import os
from nose import SkipTest
from distributions.fileutil import tempdir
import loom.generate

CLEANUP_ON_ERROR = int(os.environ.get('CLEANUP_ON_ERROR', 1))

FEATURE_TYPES = loom.generate.FEATURE_TYPES.keys()
#FEATURE_TYPES += [None]  # FIXME mixed schemas fail


def test_mixed_schema():
    if None not in FEATURE_TYPES:
        raise SkipTest('FIXME CrossCat::value_split is probably buggy')


def test_generate():
    root = os.path.abspath(os.path.curdir)
    for feature_type in FEATURE_TYPES:
        with tempdir(cleanup_on_error=CLEANUP_ON_ERROR):
            model_out = os.path.abspath('model.pb.gz')
            rows_out = os.path.abspath('rows.pbs.gz')
            os.chdir(root)
            loom.generate.generate(
                row_count=10,
                feature_count=10,
                feature_type=feature_type,
                density=0.5,
                model_out=model_out,
                rows_out=rows_out,
                debug=True,
                profile=None)
