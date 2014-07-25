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
from distributions.io.stream import (
    open_compressed,
    json_load,
    protobuf_stream_load,
)
from loom.util import mkdir_p, parallel_map, LOG
import loom
import loom.format
import loom.generate
import loom.config
import loom.runner
import parsable
parsable = parsable.Parsable()


@parsable.command
def ingest(name, schema='schema.json', rows_csv='rows.csv.gz', debug=False):
    '''
    Ingest dataset with optional json config.
    Arguments:
        name            A unique identifier for ingest + inference
        schema          Json schema file, e.g., {"feature1": "nich"}
        rows_csv        File or directory of csv files
        debug           Whether to run debug versions of C++ code
    Environment variables:
        LOOM_THREADS    Number of concurrent ingest tasks
        LOOM_VERBOSITY  Verbosity level
    '''
    if not os.path.exists(schema):
        raise IOError('Missing schema file: {}'.format(schema))
    if not os.path.exists(rows_csv):
        raise IOError('Missing rows_csv file: {}'.format(rows_csv))

    paths = loom.store.get_paths(name)
    mkdir_p(paths['root'])
    with open_compressed(paths['version'], 'w') as f:
        f.write(loom.__version__)

    LOG('making schema row')
    loom.format.make_schema_row(
        schema_in=schema,
        schema_row_out=paths['schema_row'])

    LOG('making encoding')
    loom.format.make_encoding(
        schema_in=schema,
        rows_in=rows_csv,
        encoding_out=paths['encoding'])

    LOG('importing rows')
    loom.format.import_rows(
        encoding_in=paths['encoding'],
        rows_csv_in=rows_csv,
        rows_out=paths['rows'])

    LOG('making tare rows')
    loom.runner.tare(
        schema_row_in=paths['schema_row'],
        rows_in=paths['rows'],
        tares_out=paths['tares'],
        debug=debug)

    tare_count = sum(1 for _ in protobuf_stream_load(paths['tares']))
    LOG('sparsifying rows WRT {} tare rows'.format(tare_count))
    loom.runner.sparsify(
        schema_row_in=paths['schema_row'],
        tares_in=paths['tares'],
        rows_in=paths['rows'],
        rows_out=paths['diffs'],
        debug=debug)


@parsable.command
def infer(name, sample_count=10, config=None, debug=False):
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
    assert sample_count >= 1, 'too few samples: {}'.format(sample_count)
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
    paths = loom.store.get_samples(name, sample_count=(1 + seed))[seed]

    LOG('making config')
    if config is None:
        config = {}
    elif isinstance(config, basestring):
        if not os.path.exists(config):
            raise IOError('Missing config file: {}'.format(config))
        config = json_load(config)
    else:
        config = copy.deepcopy(config)
    config['seed'] = seed
    loom.config.config_dump(config, paths['config'])

    LOG('generating init')
    loom.generate.generate_init(
        encoding_in=paths['encoding'],
        model_out=paths['init'],
        seed=seed)

    LOG('shuffling rows')
    loom.runner.shuffle(
        rows_in=paths['diffs'],
        rows_out=paths['shuffled'],
        seed=seed,
        debug=debug)

    LOG('inferring')
    loom.runner.infer(
        config_in=paths['config'],
        rows_in=paths['shuffled'],
        tares_in=paths['tares'],
        model_in=paths['init'],
        model_out=paths['model'],
        groups_out=paths['groups'],
        assign_out=paths['assign'],
        log_out=paths['infer_log'],
        debug=debug)


if __name__ == '__main__':
    parsable.dispatch()
