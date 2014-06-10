import loom.benchmark

DATASET = 'dd-10-10-0.5'


def test_shuffle():
    loom.benchmark.shuffle(DATASET, profile=None)


def test_infer():
    loom.benchmark.infer(DATASET, profile=None)


def test_checkpoint():
    loom.benchmark.load_checkpoint(DATASET, period_sec=0.2)
    loom.benchmark.infer_checkpoint(DATASET, profile=None)


def test_generate():
    loom.benchmark.generate(profile=None)
