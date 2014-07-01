# Copyright (c) 2014, Salesforce.com, Inc.  All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# - Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
# - Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
# - Neither the name of Salesforce.com nor the names of its contributors
#   may be used to endorse or promote products derived from this
#   software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
# OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
# TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
# USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
import sys
import shutil
import parsable
from distributions.fileutil import tempdir
from distributions.io.stream import open_compressed, protobuf_stream_load
from loom.util import mkdir_p, rm_rf
from loom.util import DATA
import loom.config
import loom.runner
import loom.generate
import loom.format
import loom.datasets
import loom.schema_pb2
parsable = parsable.Parsable()

CHECKPOINTS = os.path.join(DATA, 'checkpoints/{}')
RESULTS = os.path.join(DATA, 'results')


def get_results(*name_parts):
    root = os.path.join(RESULTS, *name_parts)
    mkdir_p(root)
    return {
        'root': root,
        'config': 'config.pb.gz',
        'encoding': os.path.join(root, 'encoding.json.gz'),
        'rows': os.path.join(root, 'rows.pbs.gz'),
        'shuffled': os.path.join(root, 'shuffled.pbs.gz'),
        'groups': os.path.join(root, 'groups'),
        'assign': os.path.join(root, 'assign.pbs.gz'),
        'infer_log': os.path.join(root, 'infer_log.pbs'),
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


def list_options_and_exit():
    print 'try one of:'
    for name in sorted(os.listdir(loom.datasets.DATASETS)):
        print '  {}'.format(name)
    sys.exit(1)


parsable.command(loom.runner.profilers)


@parsable.command
def shuffle(name=None, debug=False, profile='time'):
    '''
    Shuffle dataset for inference.
    '''
    if name is None:
        list_options_and_exit()

    dataset = loom.datasets.get_dataset(name)
    assert os.path.exists(dataset['rows']), 'First load dataset'
    results = get_results('shuffle', name)

    loom.runner.shuffle(
        rows_in=dataset['rows'],
        rows_out=results['shuffled'],
        debug=debug,
        profile=profile)
    assert os.path.exists(shuffled)


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
        list_options_and_exit()

    dataset = loom.datasets.get_dataset(name)
    assert os.path.exists(dataset['init']), 'First load dataset'
    assert os.path.exists(dataset['shuffled']), 'First load dataset'
    assert extra_passes > 0, 'cannot initialize with extra_passes = 0'
    results = get_results('infer', name)
    mkdir_p(results['groups'])

    config = {'schedule': {'extra_passes': extra_passes}}
    loom.config.config_dump(config, results['config'])

    loom.runner.infer(
        config_in=results['config'],
        rows_in=dataset['shuffled'],
        model_in=dataset['init'],
        groups_out=results['groups'],
        log_out=results['infer_log'],
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
        list_options_and_exit()

    dataset = loom.datasets.get_dataset(name)
    assert os.path.exists(dataset['shuffled']), 'First load dataset'
    assert os.path.exists(dataset['init']), 'First load dataset'

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
            rows_in=dataset['shuffled'],
            model_in=dataset['init'],
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
                rows_in=dataset['shuffled'],
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
        list_options_and_exit()

    dataset = loom.datasets.get_dataset(name)
    assert os.path.exists(dataset['shuffled']), 'First load dataset'
    assert os.path.exists(dataset['init']), 'First load dataset'
    checkpoint = CHECKPOINTS.format(name)
    assert os.path.exists(checkpoint), 'First load checkpoint'
    results = get_results('infer_checkpoint', name)

    config = {'schedule': {'checkpoint_period_sec': period_sec}}
    loom.config.config_dump(config, config_in)

    kwargs = {'debug': debug, 'profile': profile}
    kwargs.update(checkpoint_files(checkpoint, '_in'))

    loom.runner.infer(
        config_in=results['config'],
        rows_in=dataset['shuffled'],
        **kwargs)


@parsable.command
def ingest(name=None, debug=False, profile='time'):
    '''
    Make encoding and import rows from csv.
    '''
    if name is None:
        list_options_and_exit()

    dataset = loom.datasets.get_dataset(name)
    assert os.path.exists(dataset['rows_csv']), 'First load dataset'
    assert os.path.exists(dataset['schema']), 'First load dataset'
    results = get_results('ingest', name)

    DEVNULL = open(os.devnull, 'wb')
    loom.runner.check_call(
        command=[
            'python', '-m', 'loom.format', 'make_encoding',
            dataset['schema'], dataset['rows_csv'], results['encoding']],
        debug=debug,
        profile=profile,
        stderr=DEVNULL)
    assert os.path.exists(encoding)
    loom.runner.check_call(
        command=[
            'python', '-m', 'loom.format', 'import_rows',
            results['encoding'], dataset['rows_csv'], results['rows']],
        debug=debug,
        profile=profile,
        stderr=DEVNULL)
    assert os.path.exists(rows)


@parsable.command
def generate(
        feature_type='mixed',
        rows=10000,
        cols=100,
        density=0.5,
        debug=False,
        profile='time'):
    '''
    Generate a synthetic dataset.
    '''
    results = get_results(name)['generate']

    loom.generate.generate(
        row_count=rows,
        feature_count=cols,
        feature_type=feature_type,
        density=density,
        init_out=results['init'],
        rows_out=results['rows'],
        model_out=results['model'],
        groups_out=results['groups'],
        debug=debug,
        profile=profile)

    print 'model file is {} bytes'.format(os.path.getsize(model_out))
    print 'rows file is {} bytes'.format(os.path.getsize(rows_out))


@parsable.command
def clean():
    '''
    Clean out results.
    '''
    if os.path.exists(RESULTS):
        shutil.rmtree(RESULTS)


if __name__ == '__main__':
    parsable.dispatch()
