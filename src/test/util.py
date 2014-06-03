import os
import functools
import loom.datasets

CLEANUP_ON_ERROR = int(os.environ.get('CLEANUP_ON_ERROR', 1))

TEST_CONFIGS = [
    name
    for name, config in loom.datasets.CONFIGS.iteritems()
    if config['row_count'] <= 100
    if config['feature_count'] <= 100
]


def get_dataset(name):
    return {
        'rows': loom.datasets.ROWS.format(name),
        'model': loom.datasets.MODEL.format(name),
        'groups': loom.datasets.GROUPS.format(name),
    }


def for_each_dataset(fun):
    @functools.wraps(fun)
    def test_one(dataset):
        files = get_dataset(dataset)
        for path in files.itervalues():
            if not os.path.exists(path):
                raise ValueError(
                    'missing {}, first `python -m loom.datasets init`'.format(
                        path))
        fun(**files)

    @functools.wraps(fun)
    def test_all():
        for dataset in TEST_CONFIGS:
            yield test_one, dataset

    return test_all
