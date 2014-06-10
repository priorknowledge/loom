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
INIT = os.path.join(DATASETS, '{}/init.pb.gz')
MODEL = os.path.join(DATASETS, '{}/model.pb.gz')
GROUPS = os.path.join(DATASETS, '{}/groups')


FEATURE_TYPES = loom.schema.FEATURE_TYPES.keys()
FEATURE_TYPES += ['mixed']
COST = {
    'gp': 10,
    'mixed': 10,
}


def get_cost(config):
    cell_count = config['row_count'] * config['feature_count']
    return cell_count * COST.get(config['feature_type'], 1)


CONFIGS = [
    {
        'feature_type': feature_type,
        'row_count': row_count,
        'feature_count': feature_count,
        'density': density,
    }
    for feature_type in FEATURE_TYPES
    for row_count in [10 ** r for r in [1, 2, 3, 4, 5, 6]]
    for feature_count in [10 ** c for c in [1, 2, 3, 4]]
    for density in [0.5]
]
CONFIGS = {
    '{feature_type}-{row_count}-{feature_count}-{density}'.format(**c): c
    for c in CONFIGS
    if get_cost(c) <= 10 ** 7
}


@parsable.command
def init():
    '''
    Generate synthetic datasets for testing and benchmarking.
    '''
    configs = sorted(CONFIGS.keys(), key=(lambda c: -get_cost(CONFIGS[c])))
    parallel_map(load_one, configs)


def load_one(name):
    dataset = os.path.join(DATASETS, name)
    mkdir_p(dataset)
    init_out = INIT.format(name)
    rows_out = ROWS.format(name)
    model_out = MODEL.format(name)
    groups_out = GROUPS.format(name)
    if not all(os.path.exists(f) for f in [rows_out, model_out, groups_out]):
        print 'generating', name
        config = CONFIGS[name]
        loom.generate.generate(
            init_out=init_out,
            rows_out=rows_out,
            model_out=model_out,
            groups_out=groups_out,
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
