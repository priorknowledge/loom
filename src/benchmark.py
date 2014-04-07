import os
import shutil
import parsable
import loom.runner
import loom.format
import loom.test.util
from loom.util import parallel_map
from distributions.io.stream import json_load, protobuf_stream_load
parsable = parsable.Parsable()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data')
DATASETS = os.path.join(DATA, 'datasets')
RESULTS = os.path.join(DATA, 'results')


def mkdir_p(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


@parsable.command
def load(*names):
    '''
    Import a datasets, list available datasets, or load 'all' datasets.
    '''
    if not names:
        print 'try one of:'
        for name in loom.test.util.list_datasets():
            print '  {}'.format(name)
        return

    if names == ('all',):
        names = [
            name
            for name in loom.test.util.list_datasets()
            if not os.path.exists(os.path.join(DATASETS, name, 'rows.pbs.gz'))
        ]

    parallel_map(_load, names)


def _load(name):
    print 'loading', name
    data_path = os.path.join(DATASETS, name)
    mkdir_p(data_path)
    model = os.path.join(data_path, 'model.pb.gz')
    rows = os.path.join(data_path, 'rows.pbs.gz')

    dataset = loom.test.util.get_dataset(name)
    meta = dataset['meta']
    data = dataset['data']
    mask = dataset['mask']
    latent = dataset['latent']

    loom.format.import_latent(meta, latent, model)
    loom.format.import_data(meta, data, mask, rows)

    meta = json_load(meta)
    object_count = len(meta['object_pos'])
    feature_count = len(meta['feature_pos'])
    print '{}: {} rows x {} cols'.format(name, object_count, feature_count)


@parsable.command
def run(name=None, debug=False):
    '''
    Run loom on a dataset, or list available datasets.
    '''
    if name is None:
        print 'try one of:'
        for name in loom.test.util.list_datasets():
            if os.path.exists(os.path.join(DATASETS, name, 'rows.pbs.gz')):
                print '  {}'.format(name)
        return

    data_path = os.path.join(DATASETS, name)
    model = os.path.join(data_path, 'model.pb.gz')
    rows = os.path.join(data_path, 'rows.pbs.gz')
    groups_in = None  # TODO import groups_in
    assert os.path.exists(model), 'First load dataset'
    assert os.path.exists(rows), 'First load dataset'
    assert groups_in is None or os.path.exists(groups_in), 'First load dataset'

    results_path = os.path.join(RESULTS, name)
    mkdir_p(results_path)
    groups_out = os.path.join(results_path, 'groups_out')
    mkdir_p(groups_out)

    loom.runner.run(model, groups_in, rows, groups_out, debug=debug)

    group_counts = []
    for f in os.listdir(groups_out):
        group_count = 0
        for _ in protobuf_stream_load(os.path.join(groups_out, f)):
            group_count += 1
        group_counts.append(group_count)
    print 'group_counts: {}'.format(' '.join(map(str, group_counts)))


@parsable.command
def clean():
    '''
    Clean out data and results.
    '''
    for path in [DATASETS, RESULTS]:
        if os.path.exists(path):
            shutil.rmtree(path)


if __name__ == '__main__':
    parsable.dispatch()
