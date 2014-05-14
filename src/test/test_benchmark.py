import loom.benchmark

DATASET = 'dha'


def test_all():
    loom.benchmark.load(DATASET)
    loom.benchmark.shuffle(DATASET)
    loom.benchmark.infer(DATASET)
