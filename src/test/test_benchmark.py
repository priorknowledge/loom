import loom.benchmark

DATASET = 'dha'


def test_all():
    loom.benchmark.load(DATASET)
    loom.benchmark.info(DATASET)
    loom.benchmark.shuffle(DATASET, profile=None)
    loom.benchmark.infer(DATASET, profile=None)
    loom.benchmark.load_checkpoint(DATASET)
    loom.benchmark.infer_checkpoint(DATASET, profile=None)


def test_generate():
    for feature_type in loom.benchmark.FEATURE_TYPES:
        loom.benchmark.generate(feature_type, 10, 10, 0.5, profile=None)
