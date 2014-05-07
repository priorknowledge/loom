import os
import functools
from nose import SkipTest

try:
    import testdata
    TESTDATA_ROOT = testdata.ROOT
except ImportError:
    TESTDATA_ROOT = os.path.expanduser('~/sf/test-data/testdata/data')
    if not os.path.exists(TESTDATA_ROOT):
        TESTDATA_ROOT = os.path.expanduser('~/pk/test-data/testdata/data')


DATASETS = [
    'dha',
    'iris',
    'wine',
    'religion',
    #'census',  # very large
    #'network',  # very large
    'synth.ADD2.10.1000',
    'synth.ADD16.10.1000',
    'synth.DPM.10.1000',
    'synth.GP.10.1000',
    'synth.NICH.10.1000',
]


def list_datasets():
    sample = os.path.join('results', 'samples', 'sample_000.json')
    return sorted(
        name
        for name in os.listdir(TESTDATA_ROOT)
        if os.path.exists(os.path.join(TESTDATA_ROOT, name, sample))
    )


def get_dataset(name):
    data_root = os.path.join(TESTDATA_ROOT, name)
    samples = os.path.join(data_root, 'results', 'samples')
    return {
        'tardis_conf': os.path.join(data_root, 'conf.json'),
        'meta': os.path.join(data_root, 'dataset', 'meta.json'),
        'data': os.path.join(data_root, 'dataset', 'data.bin'),
        'mask': os.path.join(data_root, 'dataset', 'mask.bin'),
        'latent': os.path.join(samples, 'sample_000.json'),
        'predictor': os.path.join(samples, 'sample_000.predictor.json'),
    }


def for_each_dataset(fun):
    @functools.wraps(fun)
    def test_one(dataset):
        files = get_dataset(dataset)
        if not all(os.path.exists(path) for path in files.itervalues()):
            raise SkipTest('missing {}'.format(dataset))
        fun(**files)

    @functools.wraps(fun)
    def test_all():
        for dataset in DATASETS:
            yield test_one, dataset

    return test_all
