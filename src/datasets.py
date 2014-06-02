import os
import shutil
from loom.util import mkdir_p
import loom.generate
from loom.util import parallel_map
import parsable
parsable = parsable.Parsable()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data')
DATASETS = os.path.join(DATA, 'datasets')
ROWS = os.path.join(DATASETS, '{}/rows.pbs.gz')
MODEL = os.path.join(DATASETS, '{}/model.pb.gz')


FEATURE_TYPES = loom.schema.FEATURE_TYPES.keys()
FEATURE_TYPES += ['mixed']

CONFIGS = [
    {
        'feature_type': feature_type,
        'row_count': row_count,
        'feature_count': feature_count,
        'density': density,
    }
    for feature_type in FEATURE_TYPES
    for row_count in [10 ** r for r in [2, 3, 4, 5, 6]]
    for feature_count in [10 ** c for c in [1, 2, 3, 4]]
    if row_count * feature_count <= 10 ** 7
    for density in [0.5]
]
CONFIGS = {
    '{feature_type}-{row_count}-{feature_count}-{density}'.format(**c): c
    for c in CONFIGS
}


@parsable.command
def init():
    '''
    Generate synthetic datasets for testing and benchmarking.
    '''
    parallel_map(load_one, CONFIGS.keys())


def load_one(name):
    dataset = os.path.join(DATASETS, name)
    mkdir_p(dataset)
    model_out = MODEL.format(name)
    rows_out = ROWS.format(name)
    if not all(os.path.exists(f) for f in [model_out, rows_out]):
        print 'generating', name
        config = CONFIGS[name]
        loom.generate.generate(
            model_out=model_out,
            rows_out=rows_out,
            **config)


@parsable.command
def clean():
    '''
    Clean out datasets.
    '''
    if os.path.exists(DATASETS):
        shutil.rmtree(DATASETS)


if __name__ == '__main__':
    parsable.dispatch()
