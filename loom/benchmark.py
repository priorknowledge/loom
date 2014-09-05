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
import glob
import parsable
from distributions.io.stream import (
    open_compressed,
    json_load,
    protobuf_stream_load,
)
from loom.util import chdir, mkdir_p, rm_rf
import loom.store
import loom.config
import loom.runner
import loom.generate
import loom.format
import loom.datasets
import loom.schema_pb2
import loom.query
import loom.preql
parsable = parsable.Parsable()


def get_paths(name, operation):
    inputs = loom.store.get_paths(name)
    results = loom.store.get_paths(os.path.join(name, 'benchmark', operation))
    return inputs, results


def checkpoint_files(path, suffix=''):
    path = os.path.abspath(str(path))
    assert os.path.exists(path), path
    return {
        'model' + suffix: os.path.join(path, 'model.pb.gz'),
        'groups' + suffix: os.path.join(path, 'groups'),
        'assign' + suffix: os.path.join(path, 'assign.pbs.gz'),
        'checkpoint' + suffix: os.path.join(path, 'checkpoint.pb.gz'),
    }


parsable.command(loom.runner.profilers)


@parsable.command
def generate(
        feature_type='mixed',
        row_count=10000,
        feature_count=100,
        density=0.5,
        debug=False,
        profile='time'):
    '''
    Generate a synthetic dataset.
    '''
    name = '{}-{}-{}-{}'.format(
        feature_type,
        row_count,
        feature_count,
        density)
    inputs, results = get_paths(name, 'generate')

    loom.generate.generate(
        row_count=row_count,
        feature_count=feature_count,
        feature_type=feature_type,
        density=density,
        init_out=results['samples'][0]['init'],
        rows_out=results['ingest']['rows'],
        model_out=results['samples'][0]['model'],
        groups_out=results['samples'][0]['groups'],
        debug=debug,
        profile=profile)
    loom.format.make_schema(
        model_in=results['samples'][0]['model'],
        schema_out=results['ingest']['schema'])
    loom.format.make_fake_encoding(
        schema_in=results['ingest']['schema'],
        model_in=results['samples'][0]['model'],
        encoding_out=results['ingest']['encoding'])
    loom.format.make_schema_row(
        schema_in=results['ingest']['schema'],
        schema_row_out=results['ingest']['schema_row'])

    loom.store.provide(name, results, [
        'ingest.rows',
        'ingest.schema',
        'ingest.schema_row',
        'ingest.encoding',
        'samples.0.init',
        'samples.0.model',
        'samples.0.groups',
    ])

    return name


@parsable.command
def ingest(name=None, debug=False, profile='time'):
    '''
    Make encoding and import rows from csv.
    '''
    loom.store.require(name, ['ingest.schema', 'ingest.rows_csv'])
    inputs, results = get_paths(name, 'ingest')

    DEVNULL = open(os.devnull, 'wb')
    loom.runner.check_call(
        command=[
            'python', '-m', 'loom.format', 'make_encoding',
            inputs['ingest']['schema'],
            inputs['ingest']['rows_csv'],
            results['ingest']['encoding']],
        debug=debug,
        profile=profile,
        stderr=DEVNULL)
    loom.store.provide(name, results, ['ingest.encoding'])

    loom.runner.check_call(
        command=[
            'python', '-m', 'loom.format', 'import_rows',
            inputs['ingest']['encoding'],
            inputs['ingest']['rows_csv'],
            results['ingest']['rows']],
        debug=debug,
        profile=profile,
        stderr=DEVNULL)
    loom.store.provide(name, results, ['ingest.rows'])


@parsable.command
def tare(name=None, debug=False, profile='time'):
    '''
    Find tare rows.
    '''
    loom.store.require(name, ['ingest.rows', 'ingest.schema_row'])
    inputs, results = get_paths(name, 'tare')

    loom.runner.tare(
        schema_row_in=inputs['ingest']['schema_row'],
        rows_in=inputs['ingest']['rows'],
        tares_out=results['ingest']['tares'],
        debug=debug,
        profile=profile)

    loom.store.provide(name, results, ['ingest.tares'])


@parsable.command
def sparsify(name=None, debug=False, profile='time'):
    '''
    Sparsify dataset WRT tare rows.
    '''
    loom.store.require(name, [
        'ingest.rows',
        'ingest.schema_row',
        'ingest.tares',
    ])
    inputs, results = get_paths(name, 'sparsify')

    loom.runner.sparsify(
        schema_row_in=inputs['ingest']['schema_row'],
        tares_in=inputs['ingest']['tares'],
        rows_in=inputs['ingest']['rows'],
        rows_out=results['ingest']['diffs'],
        debug=debug,
        profile=profile)

    loom.store.provide(name, results, ['ingest.diffs'])


@parsable.command
def init(name=None):
    '''
    Generate initial model for inference.
    '''
    loom.store.require(name, ['ingest.encoding'])
    inputs, results = get_paths(name, 'init')

    loom.generate.generate_init(
        encoding_in=inputs['ingest']['encoding'],
        model_out=results['samples'][0]['init'])

    loom.store.provide(name, results, ['samples.0.init'])


@parsable.command
def shuffle(name=None, debug=False, profile='time'):
    '''
    Shuffle dataset for inference.
    '''
    loom.store.require(name, ['ingest.diffs'])
    inputs, results = get_paths(name, 'shuffle')

    loom.runner.shuffle(
        rows_in=inputs['ingest']['diffs'],
        rows_out=results['samples'][0]['shuffled'],
        debug=debug,
        profile=profile)

    loom.store.provide(name, results, ['samples.0.shuffled'])


@parsable.command
def infer(
        name=None,
        extra_passes=loom.config.DEFAULTS['schedule']['extra_passes'],
        parallel=True,
        debug=False,
        profile='time'):
    '''
    Run inference on a dataset, or list available datasets.
    '''
    assert extra_passes > 0, 'cannot initialize with extra_passes = 0'
    loom.store.require(name, ['samples.0.init', 'samples.0.shuffled'])
    inputs, results = get_paths(name, 'infer')

    config = {'schedule': {'extra_passes': extra_passes}}
    if not parallel:
        loom.config.fill_in_sequential(config)
    loom.config.config_dump(config, results['samples'][0]['config'])

    loom.runner.infer(
        config_in=results['samples'][0]['config'],
        rows_in=inputs['samples'][0]['shuffled'],
        tares_in=inputs['ingest']['tares'],
        model_in=inputs['samples'][0]['init'],
        model_out=results['samples'][0]['model'],
        groups_out=results['samples'][0]['groups'],
        log_out=results['samples'][0]['infer_log'],
        debug=debug,
        profile=profile)

    loom.store.provide(name, results, [
        'samples.0.config',
        'samples.0.model',
        'samples.0.groups',
    ])

    groups = results['samples'][0]['groups']
    assert os.listdir(groups), 'no groups were written'
    group_counts = []
    for f in os.listdir(groups):
        group_count = 0
        for _ in protobuf_stream_load(os.path.join(groups, f)):
            group_count += 1
        group_counts.append(group_count)
    print 'group_counts: {}'.format(' '.join(map(str, group_counts)))


def _load_checkpoint(step):
    message = loom.schema_pb2.Checkpoint()
    filename = checkpoint_files(step)['checkpoint']
    with open_compressed(filename) as f:
        message.ParseFromString(f.read())
    return message


@parsable.command
def load_checkpoint(name=None, period_sec=5, debug=False):
    '''
    Grab last full checkpoint for profiling, or list available datasets.
    '''
    loom.store.require(name, ['samples.0.init', 'samples.0.shuffled'])
    inputs, results = get_paths(name, 'checkpoints')

    rm_rf(results['root'])
    mkdir_p(results['root'])
    with chdir(results['root']):

        config = {'schedule': {'checkpoint_period_sec': period_sec}}
        loom.config.config_dump(config, results['samples'][0]['config'])

        # run first iteration
        step = 0
        mkdir_p(str(step))
        kwargs = checkpoint_files(step, '_out')
        print 'running checkpoint {}, tardis_iter 0'.format(step)
        loom.runner.infer(
            config_in=results['samples'][0]['config'],
            rows_in=inputs['samples'][0]['shuffled'],
            tares_in=inputs['ingest']['tares'],
            model_in=inputs['samples'][0]['init'],
            log_out=results['samples'][0]['infer_log'],
            debug=debug,
            **kwargs)
        checkpoint = _load_checkpoint(step)

        # find penultimate checkpoint
        while not checkpoint.finished:
            rm_rf(str(step - 3))
            step += 1
            print 'running checkpoint {}, tardis_iter {}'.format(
                step,
                checkpoint.tardis_iter)
            kwargs = checkpoint_files(step - 1, '_in')
            mkdir_p(str(step))
            kwargs.update(checkpoint_files(step, '_out'))
            loom.runner.infer(
                config_in=results['samples'][0]['config'],
                rows_in=inputs['samples'][0]['shuffled'],
                tares_in=inputs['ingest']['tares'],
                log_out=results['samples'][0]['infer_log'],
                debug=debug,
                **kwargs)
            checkpoint = _load_checkpoint(step)

        print 'final checkpoint {}, tardis_iter {}'.format(
            step,
            checkpoint.tardis_iter)

        last_full = str(step - 2)
        assert os.path.exists(last_full), 'too few checkpoints'
        checkpoint = _load_checkpoint(step)
        print 'saving checkpoint {}, tardis_iter {}'.format(
            last_full,
            checkpoint.tardis_iter)
        for f in checkpoint_files(last_full).values():
            shutil.move(f, results['root'])
        for f in glob.glob(os.path.join(results['root'], '[0-9]*/')):
            shutil.rmtree(f)


@parsable.command
def infer_checkpoint(
        name=None,
        period_sec=0,
        parallel=True,
        debug=False,
        profile='time'):
    '''
    Run inference from checkpoint, or list available checkpoints.
    '''
    loom.store.require(name, ['samples.0.init', 'samples.0.shuffled'])
    checkpoint = get_paths(name, 'checkpoints')[1]['root']
    assert os.path.exists(checkpoint), 'First load checkpoint'
    inputs, results = get_paths(name, 'infer_checkpoint')

    config = {'schedule': {'checkpoint_period_sec': period_sec}}
    if not parallel:
        loom.config.fill_in_sequential(config)
    loom.config.config_dump(config, results['samples'][0]['config'])

    kwargs = {'debug': debug, 'profile': profile}
    kwargs.update(checkpoint_files(checkpoint, '_in'))

    loom.runner.infer(
        config_in=results['samples'][0]['config'],
        rows_in=inputs['samples'][0]['shuffled'],
        tares_in=inputs['ingest']['tares'],
        log_out=results['samples'][0]['infer_log'],
        **kwargs)


@parsable.command
def related(
        name=None,
        sample_count=loom.preql.SAMPLE_COUNT,
        debug=False,
        profile='time'):
    '''
    Run related query.
    '''
    loom.store.require(name, [
        'ingest.schema',
        'ingest.encoding',
        'samples.0.config',
        'samples.0.model',
        'samples.0.groups',
    ])
    inputs, results = get_paths(name, 'related')
    root = inputs['root']
    encoding = inputs['ingest']['encoding']
    features = sorted(json_load(inputs['ingest']['schema']).keys())

    with loom.preql.get_server(root, encoding, debug, profile) as preql:
        print 'querying {} features'.format(len(features)),
        for feature in features:
            preql.relate([feature], sample_count=sample_count)
            sys.stdout.write('.')
            sys.stdout.flush()
        print '\ndone'


@parsable.command
def test(name=None, debug=True, profile=None):
    '''
    Run pipeline: tare; sparsify; init; shuffle; infer; related.
    '''
    loom.store.require(name, ['ingest.rows', 'ingest.schema_row'])
    options = dict(debug=debug, profile=profile)
    tare(name, **options)
    sparsify(name, **options)
    init(name)
    shuffle(name, **options)
    infer(name, **options)
    related(name, **options)


if __name__ == '__main__':
    parsable.dispatch()
