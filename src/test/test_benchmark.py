import loom.benchmark
from loom.test.test_generate import FEATURE_TYPES

DATASET = 'dha'


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
            profile=None)
