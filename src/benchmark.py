import os
import parsable
import loom.runner
import loom.format
import loom.test.util
from distributions.fileutil import json_load, protobuf_stream_load
parsable = parsable.Parsable()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data')
DATASETS = os.path.join(DATA, 'datasets')
RESULTS = os.path.join(DATA, 'results')


def mkdir_p(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


@parsable.command
def load(name=None):
    '''
    Import a dataset, or list available datasets.
    '''
    if name is None:
        print 'try one of:'
        for name in loom.test.util.list_datasets():
            print '  {}'.format(name)
        return

    data_path = os.path.join(DATASETS, name)
    mkdir_p(data_path)
    model = os.path.join(data_path, 'model.pb')
    values = os.path.join(data_path, 'rows.pbs.gz')

    dataset = loom.test.util.get_dataset(name)
    meta = dataset['meta']
    data = dataset['data']
    mask = dataset['mask']
    latent = dataset['latent']

    loom.format.import_latent(meta, latent, model)
    loom.format.import_data(meta, data, mask, values)

    meta = json_load(meta)
    object_count = len(meta['object_pos'])
    feature_count = len(meta['feature_pos'])
    print '{}: {} rows x {} cols'.format(name, object_count, feature_count)


@parsable.command
def run(name):
    '''
    Run loom on a dataset.
    '''
    data_path = os.path.join(DATASETS, name)
    model = os.path.join(data_path, 'model.pb')
    values = os.path.join(data_path, 'rows.pbs.gz')
    assert os.path.exists(model), 'First load dataset'
    assert os.path.exists(values), 'First load dataset'

    results_path = os.path.join(RESULTS, name)
    mkdir_p(results_path)
    groups = os.path.join(results_path, 'groups')
    mkdir_p(groups)

    loom.runner.run(model, values, groups, safe=False)

    group_counts = []
    for f in os.listdir(groups):
        group_count = 0
        for _ in protobuf_stream_load(os.path.join(groups, f)):
            group_count += 1
        group_counts.append(group_count)
    print 'group_counts: {}'.format(' '.join(map(str, group_counts)))


if __name__ == '__main__':
    parsable.dispatch()
