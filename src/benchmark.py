import os
import sys
import shutil
import numpy
import parsable
import loom.config
import loom.runner
import loom.format
import loom.cFormat
import loom.schema_pb2
import loom.test.util
from loom.util import parallel_map
from distributions.fileutil import tempdir
from distributions.io.stream import (
    open_compressed,
    json_load,
    protobuf_stream_load,
)
from distributions.lp.models import bb, dd, dpd, nich, gp
from distributions.lp.clustering import PitmanYor
parsable = parsable.Parsable()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, 'data')
DATASETS = os.path.join(DATA, 'datasets')
CHECKPOINTS = os.path.join(DATA, 'checkpoints/{}')
RESULTS = os.path.join(DATA, 'results')
ROWS = os.path.join(DATASETS, '{}/rows.pbs.gz')
MODEL = os.path.join(DATASETS, '{}/model.pb.gz')
GROUPS = os.path.join(DATASETS, '{}/groups')
ASSIGN = os.path.join(DATASETS, '{}/assign.pbs.gz')

CLUSTERING = PitmanYor.from_dict({'alpha': 2.0, 'd': 0.1})
FEATURE_TYPES = {
    'bb': bb,
    'dd': dd,
    'dpd': dpd,
    'gp': gp,
    'nich': nich,
}


def checkpoint_files(path, suffix=''):
    path = os.path.abspath(str(path))
    assert os.path.exists(path), path
    return {
        'model' + suffix: os.path.join(path, 'model.pb.gz'),
        'groups' + suffix: os.path.join(path, 'groups'),
        'assign' + suffix: os.path.join(path, 'assign.pbs.gz'),
        'checkpoint' + suffix: os.path.join(path, 'checkpoint.pb.gz'),
    }


def mkdir_p(dirname):
    'like mkdir -p'
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def rm_rf(dirname):
    'like rm -rf'
    if os.path.exists(dirname):
        shutil.rmtree(dirname)


def list_options_and_exit(*required):
    print 'try one of:'
    for name in loom.test.util.list_datasets():
        if all(os.path.exists(patt.format(name)) for patt in required):
            print '  {}'.format(name)
    sys.exit(1)


parsable.command(loom.runner.profilers)


@parsable.command
def load(name=None, debug=False):
    '''
    Import a datasets, list available datasets, or load 'all' datasets.
    '''
    if name is None:
        list_options_and_exit()

    names = [name]

    if names == ['all']:
        names = [
            n
            for n in loom.test.util.list_datasets()
            if not os.path.exists(ROWS.format(n))
        ]

    args = [(n, debug) for n in names]
    parallel_map(_load, args)


def _load((name, debug)):
    print 'loading', name
    data_path = os.path.join(DATASETS, name)
    mkdir_p(data_path)
    model = MODEL.format(name)
    groups = GROUPS.format(name)
    assign = ASSIGN.format(name)
    rows = ROWS.format(name)

    dataset = loom.test.util.get_dataset(name)
    meta = dataset['meta']
    data = dataset['data']
    mask = dataset['mask']
    latent = dataset['latent']
    tardis_conf = dataset['tardis_conf']

    loom.format.import_latent(
        meta_in=meta,
        latent_in=latent,
        tardis_conf_in=tardis_conf,
        model_out=model,
        groups_out=groups,
        assign_out=assign)
    loom.format.import_data(
        meta_in=meta,
        data_in=data,
        mask_in=mask,
        rows_out=rows,
        validate=debug)
    loom.runner.shuffle(rows_in=rows, rows_out=rows)

    meta = json_load(meta)
    object_count = len(meta['object_pos'])
    feature_count = len(meta['feature_pos'])
    print '{}: {} rows x {} cols'.format(name, object_count, feature_count)


@parsable.command
def info(name=None, debug=False):
    '''
    Get information about a dataset, or list available datasets.
    '''
    if name is None:
        list_options_and_exit(ROWS)

    if debug:
        pos = 0
        dumped = 'None'
        sizes = []
        try:
            rows = loom.cFormat.protobuf_stream_load(ROWS.format(name))
            for pos, row in enumerate(rows):
                dumped = row.dump()
                sizes.append(row.ByteSize())
        except:
            print 'error after row {} with data:\n{}'.format(pos, dumped)
            raise
    else:
        rows = loom.cFormat.protobuf_stream_load(ROWS.format(name))
        sizes = [row.ByteSize() for row in rows]

    print 'row count:\t{}'.format(len(sizes))
    print 'min bytes:\t{}'.format(min(sizes))
    print 'mean bytes:\t{}'.format(numpy.mean(sizes))
    print 'max bytes:\t{}'.format(max(sizes))


@parsable.command
def shuffle(name=None, debug=False, profile='time'):
    '''
    Shuffle dataset for inference.
    '''
    if name is None:
        list_options_and_exit(ROWS)

    rows_in = ROWS.format(name)
    assert os.path.exists(rows_in), 'First load dataset'

    destin = os.path.join(RESULTS, name)
    mkdir_p(destin)
    rows_out = os.path.join(destin, 'rows.pbs.gz')

    loom.runner.shuffle(
        rows_in=rows_in,
        rows_out=rows_out,
        debug=debug,
        profile=profile)
    assert os.path.exists(rows_out)


@parsable.command
def infer(
        name=None,
        extra_passes=loom.config.DEFAULTS['schedule']['extra_passes'],
        debug=False,
        profile='time'):
    '''
    Run inference on a dataset, or list available datasets.
    '''
    if name is None:
        list_options_and_exit(ROWS)

    model = MODEL.format(name)
    rows = ROWS.format(name)
    assert os.path.exists(model), 'First load dataset'
    assert os.path.exists(rows), 'First load dataset'

    if extra_passes > 0:
        print 'Learning structure from scratch'
        groups_in = None
    else:
        print 'Priming structure with known groups'
        groups_in = GROUPS.format(name)
        assert os.path.exists(groups_in), 'First load dataset'

    destin = os.path.join(RESULTS, name)
    mkdir_p(destin)
    groups_out = os.path.join(destin, 'groups')
    mkdir_p(groups_out)

    config = {'schedule': {'extra_passes': extra_passes}}
    config_in = os.path.join(destin, 'config.pb.gz')
    loom.config.config_dump(config, config_in)

    loom.runner.infer(
        config_in=config_in,
        rows_in=rows,
        model_in=model,
        groups_in=groups_in,
        groups_out=groups_out,
        debug=debug,
        profile=profile)

    assert os.listdir(groups_out), 'no groups were written'
    group_counts = []
    for f in os.listdir(groups_out):
        group_count = 0
        for _ in protobuf_stream_load(os.path.join(groups_out, f)):
            group_count += 1
        group_counts.append(group_count)
    print 'group_counts: {}'.format(' '.join(map(str, group_counts)))


@parsable.command
def load_checkpoint(name=None, period_sec=5, debug=False):
    '''
    Grab last full checkpoint for profiling, or list available datasets.
    '''
    if name is None:
        list_options_and_exit(MODEL)

    rows = ROWS.format(name)
    model = MODEL.format(name)
    assert os.path.exists(model), 'First load dataset'
    assert os.path.exists(rows), 'First load dataset'

    destin = CHECKPOINTS.format(name)
    rm_rf(destin)
    mkdir_p(os.path.dirname(destin))

    def load_checkpoint(name):
        checkpoint = loom.schema_pb2.Checkpoint()
        with open_compressed(checkpoint_files(name)['checkpoint']) as f:
            checkpoint.ParseFromString(f.read())
        return checkpoint

    with tempdir(cleanup_on_error=(not debug)):

        config = {'schedule': {'checkpoint_period_sec': period_sec}}
        config_in = os.path.abspath('config.pb.gz')
        loom.config.config_dump(config, config_in)

        # run first iteration
        step = 0
        mkdir_p(str(step))
        kwargs = checkpoint_files(str(step), '_out')
        print 'running checkpoint {}, tardis_iter 0'.format(step)
        loom.runner.infer(
            config_in=config_in,
            rows_in=rows,
            model_in=model,
            debug=debug,
            **kwargs)
        checkpoint = load_checkpoint(step)

        # find penultimate checkpoint
        while not checkpoint.finished:
            rm_rf(str(step - 3))
            step += 1
            print 'running checkpoint {}, tarids_iter {}'.format(
                step,
                checkpoint.tardis_iter)
            kwargs = checkpoint_files(step - 1, '_in')
            mkdir_p(str(step))
            kwargs.update(checkpoint_files(step, '_out'))
            loom.runner.infer(
                config_in=config_in,
                rows_in=rows,
                debug=debug,
                **kwargs)
            checkpoint = load_checkpoint(step)

        print 'final checkpoint {}, tardis_iter {}'.format(
            step,
            checkpoint.tardis_iter)

        last_full = str(step - 2)
        assert os.path.exists(last_full), 'too few checkpoints'
        checkpoint = load_checkpoint(step)
        print 'saving checkpoint {}, tardis_iter {}'.format(
            last_full,
            checkpoint.tardis_iter)
        shutil.move(last_full, destin)


@parsable.command
def infer_checkpoint(name=None, period_sec=0, debug=False, profile='time'):
    '''
    Run inference from checkpoint, or list available checkpoints.
    '''
    if name is None:
        list_options_and_exit(CHECKPOINTS)

    rows = ROWS.format(name)
    model = MODEL.format(name)
    checkpoint = CHECKPOINTS.format(name)
    assert os.path.exists(rows), 'First load dataset'
    assert os.path.exists(model), 'First load dataset'
    assert os.path.exists(checkpoint), 'First load checkpoint'

    destin = os.path.join(RESULTS, name)
    mkdir_p(destin)

    config = {'schedule': {'checkpoint_period_sec': period_sec}}
    config_in = os.path.join(destin, 'config.pb.gz')
    loom.config.config_dump(config, config_in)

    kwargs = {'debug': debug, 'profile': profile}
    kwargs.update(checkpoint_files(checkpoint, '_in'))

    loom.runner.infer(config_in=config_in, rows_in=rows, **kwargs)


def generate_model(feature_type, feature_count):
    module = FEATURE_TYPES[feature_type]
    shared = module.Shared.from_dict(module.EXAMPLES[-1]['shared'])
    cross_cat = loom.schema_pb2.CrossCat()
    for featureid in xrange(feature_count):
        kind = cross_cat.kinds.add()
        CLUSTERING.dump_protobuf(kind.product_model.clustering.pitman_yor)
        features = getattr(kind.product_model, feature_type)
        shared.dump_protobuf(features.add())
        kind.featureids.append(featureid)
        cross_cat.featureid_to_kindid.append(featureid)
    CLUSTERING.dump_protobuf(cross_cat.feature_clustering.pitman_yor)
    return cross_cat


@parsable.command
def generate(
        feature_type='dd',
        rows=10000,
        cols=100,
        density=0.5,
        debug=False,
        profile='time'):
    '''
    Generate a synthetic dataset.
    '''
    root = os.path.abspath(os.path.curdir)
    with tempdir(cleanup_on_error=(not debug)):
        model = generate_model(feature_type, cols)
        model_in = os.path.abspath('model.pb.gz')
        with open_compressed(model_in, 'w') as f:
            f.write(model.SerializeToString())

        config = {'generate': {'row_count': rows, 'density': density}}
        config_in = os.path.abspath('config.pb.gz')
        loom.config.config_dump(config, config_in)

        rows_out = os.path.abspath('rows.pbs.gz')

        os.chdir(root)
        loom.runner.generate(
            config_in,
            model_in,
            rows_out,
            debug=debug,
            profile=profile)


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
