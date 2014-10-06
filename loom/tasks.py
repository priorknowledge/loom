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
import copy
from distributions.io.stream import json_load
from distributions.io.stream import open_compressed
from distributions.io.stream import protobuf_stream_load
from loom.util import LOG
from loom.util import LoomError
from loom.util import parallel_map
import loom
import loom.format
import loom.generate
import loom.config
import loom.consensus
import loom.runner
import loom.preql
import loom.documented
import parsable
parsable = parsable.Parsable()

DEFAULTS = {
    'sample_count': 10,
}


@parsable.command
def ingest(
        name,
        schema='schema.json',
        rows_csv='rows.csv.gz',
        id_field=None,
        debug=False):
    '''
    Ingest dataset with optional json config.
    Arguments:
        name            A unique identifier for ingest + inference
        schema          Json schema file, e.g., {"feature1": "nich"}
        rows_csv        File or directory of csv files or csv.gz files
        id_field        Column name of id field in input csv
        debug           Whether to run debug versions of C++ code
    Environment variables:
        LOOM_THREADS    Number of concurrent ingest tasks
        LOOM_VERBOSITY  Verbosity level
    '''
    if not os.path.exists(schema):
        raise LoomError('Missing schema file: {}'.format(schema))
    if not os.path.exists(rows_csv):
        raise LoomError('Missing rows_csv file: {}'.format(rows_csv))

    paths = loom.store.get_paths(name)
    with open_compressed(paths['ingest']['version'], 'w') as f:
        f.write(loom.__version__)

    LOG('making schema row')
    loom.format.make_schema_row(
        schema_in=schema,
        schema_row_out=paths['ingest']['schema_row'])

    LOG('making encoding')
    loom.format.make_encoding(
        schema_in=schema,
        rows_in=rows_csv,
        encoding_out=paths['ingest']['encoding'])

    LOG('importing rows')
    loom.format.import_rows(
        encoding_in=paths['ingest']['encoding'],
        rows_csv_in=rows_csv,
        rows_out=paths['ingest']['rows'])

    LOG('importing rowids')
    loom.format.import_rowids(
        rows_csv_in=rows_csv,
        rowids_out=paths['ingest']['rowids'],
        id_field=id_field)

    LOG('making tare rows')
    loom.runner.tare(
        schema_row_in=paths['ingest']['schema_row'],
        rows_in=paths['ingest']['rows'],
        tares_out=paths['ingest']['tares'],
        debug=debug)

    tare_count = sum(1 for _ in protobuf_stream_load(paths['ingest']['tares']))
    LOG('sparsifying rows WRT {} tare rows'.format(tare_count))
    loom.runner.sparsify(
        schema_row_in=paths['ingest']['schema_row'],
        tares_in=paths['ingest']['tares'],
        rows_in=paths['ingest']['rows'],
        rows_out=paths['ingest']['diffs'],
        debug=debug)
    loom.config.config_dump({}, paths['query']['config'])


@parsable.command
def infer(
        name,
        sample_count=DEFAULTS['sample_count'],
        config=None,
        debug=False):
    '''
    Infer samples in parallel.
    Arguments:
        name            A unique identifier for ingest + inference
        sample_count    The number of samples to draw, typically 10-100
        config          An optional json config file, e.g.,
                            {"schedule": {"extra_passes": 500.0}}
        debug           Whether to run debug versions of C++ code
    Environment variables:
        LOOM_THREADS    Number of concurrent inference tasks
        LOOM_VERBOSITY  Verbosity level
    '''
    if not (sample_count >= 1):
        raise LoomError('Too few samples: {}'.format(sample_count))
    parallel_map(_infer_one, [
        (name, seed, config, debug) for seed in xrange(sample_count)
    ])


def _infer_one(args):
    infer_one(*args)


@parsable.command
def infer_one(name, seed=0, config=None, debug=False):
    '''
    Infer a single sample.
    Arguments:
        name            A unique identifier for ingest + inference
        seed            The seed, i.e., sample number typically 0-9
        config          An optional json config file, e.g.,
                            {"schedule": {"extra_passes": 500.0}}
        debug           Whether to run debug versions of C++ code
    Environment variables:
        LOOM_VERBOSITY  Verbosity level
    '''
    paths = loom.store.get_paths(name, sample_count=(1 + seed))
    sample = paths['samples'][seed]

    LOG('making config')
    if config is None:
        config = {}
    elif isinstance(config, basestring):
        if not os.path.exists(config):
            raise LoomError('Missing config file: {}'.format(config))
        config = json_load(config)
    else:
        config = copy.deepcopy(config)
    config['seed'] = seed
    loom.config.config_dump(config, sample['config'])

    LOG('generating init')
    loom.generate.generate_init(
        encoding_in=paths['ingest']['encoding'],
        model_out=sample['init'],
        seed=seed)

    LOG('shuffling rows')
    loom.runner.shuffle(
        rows_in=paths['ingest']['diffs'],
        rows_out=sample['shuffled'],
        seed=seed,
        debug=debug)

    LOG('inferring, watch {}'.format(sample['infer_log']))
    loom.runner.infer(
        config_in=sample['config'],
        rows_in=sample['shuffled'],
        tares_in=paths['ingest']['tares'],
        model_in=sample['init'],
        model_out=sample['model'],
        groups_out=sample['groups'],
        assign_out=sample['assign'],
        log_out=sample['infer_log'],
        debug=debug)


@parsable.command
def make_consensus(name, config=None, debug=False):
    '''
    Combine samples into a single consensus sample.
    Arguments:
        name            A unique identifier for consensus
        config          An optional json config file
                            currently doesn't do anything but will be used to
                            support e.g. cluster coarseness in the future
        debug           Whether to run debug versions of C++ code
    Environment varibles:
        LOOM_VERBOSITY  Verbosity level
    '''
    paths = loom.store.get_paths(name)
    LOG('making config')
    if config is None:
        config = {}
    elif isinstance(config, basestring):
        if not os.path.exists(config):
            raise LoomError('Missing config file: {}'.format(config))
        config = json_load(config)
    else:
        config = copy.deepcopy(config)
    loom.config.config_dump(config, paths['samples'][0]['config'])

    LOG('finding consensus')
    loom.consensus.make_consensus(paths=paths, debug=debug)


@loom.documented.transform(
    inputs=[
        'ingest.encoding',
        'samples.0.config',
        'samples.0.model',
        'samples.0.groups'])
def query(name, config=None, debug=False, profile=None):
    '''
    Start the query server.
    Arguments:
        name            A unique identifier for ingest + inference
        config          An optional json config file
        debug           Whether to run debug versions of C++ code
    Environment varibles:
        LOOM_VERBOSITY  Verbosity level
    '''
    paths = loom.store.get_paths(name)
    LOG('starting query server')
    server = loom.preql.get_server(
        paths['root'],
        paths['ingest']['encoding'],
        config=config,
        debug=debug,
        profile=profile)
    return server


if __name__ == '__main__':
    parsable.dispatch()
