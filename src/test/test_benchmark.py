import loom.benchmark

DATASET = 'dha'


def test_all():
    loom.benchmark.load(DATASET)
    loom.benchmark.info(DATASET)
    loom.benchmark.shuffle(DATASET, profile=None)
    loom.benchmark.infer(DATASET, profile=None)
    loom.benchmark.load_checkpoint(DATASET)
    loom.benchmark.infer_checkpoint(DATASET, profile=None)
