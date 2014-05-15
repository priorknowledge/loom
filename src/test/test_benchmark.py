import loom.benchmark

DATASET = 'dha'


def test_all():
    loom.benchmark.load(DATASET)
    loom.benchmark.shuffle(DATASET, profile=None)
    loom.benchmark.infer(DATASET, profile=None)
