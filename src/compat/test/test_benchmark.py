import loom.compat.benchmark

DATASET = 'dha'


def test_all():
    loom.compat.benchmark.load(DATASET)
    loom.compat.benchmark.info(DATASET)
    loom.compat.benchmark.shuffle(DATASET, profile=None)
    loom.compat.benchmark.infer(DATASET, profile=None)
    loom.compat.benchmark.load_checkpoint(DATASET)
    loom.compat.benchmark.infer_checkpoint(DATASET, profile=None)
