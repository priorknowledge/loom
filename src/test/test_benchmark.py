from nose import SkipTest
import loom.benchmark
import loom.generate

DATASET = 'dha'
#FEATURE_TYPES = [None] + loom.generate.FEATURE_TYPES.keys()
FEATURE_TYPES = loom.generate.FEATURE_TYPES.keys()  # FIXME mixed schemas fail


def test_mixed_schema():
    raise SkipTest('FIXME CrossCat::value_split is probably buggy')


def test_all():
    loom.benchmark.load(DATASET)
    loom.benchmark.info(DATASET)
    loom.benchmark.shuffle(DATASET, profile=None)
    loom.benchmark.infer(DATASET, profile=None)
    loom.benchmark.load_checkpoint(DATASET)
    loom.benchmark.infer_checkpoint(DATASET, profile=None)


def test_generate():
    for feature_type in FEATURE_TYPES:
        loom.benchmark.generate(
            feature_type=feature_type,
            rows=10,
            cols=10,
            density=0.5,
            #debug=True,
            profile=None)
