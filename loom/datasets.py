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
from distributions.io.stream import json_load, json_dump
from loom.util import mkdir_p
import loom.generate
import loom.format
from loom.util import parallel_map
import loom.store
import parsable
parsable = parsable.Parsable()


FEATURE_TYPES = loom.schema.FEATURE_TYPES.keys()
FEATURE_TYPES += ['mixed']
COST = {
    'gp': 10,
    'mixed': 10,
}


def get_cost(config):
    cell_count = (
        config['row_count'] *
        config['feature_count'] *
        config['density'])
    return cell_count * COST.get(config['feature_type'], 1)


CONFIG_VALUES = [
    {
        'feature_type': feature_type,
        'row_count': row_count,
        'feature_count': feature_count,
        'density': density,
    }
    for feature_type in FEATURE_TYPES
    for row_count in [10 ** r for r in [1, 2, 3, 4, 5, 6]]
    for feature_count in [10 ** c for c in [1, 2, 3, 4]]
    for density in [0.05, 0.5]
    if density * feature_count > 1
]
CONFIGS = {
    '{feature_type}-{row_count}-{feature_count}-{density}'.format(**c): c
    for c in CONFIG_VALUES
    if get_cost(c) <= 10 ** 7
}


@parsable.command
def generate():
    '''
    Generate synthetic datasets for testing and benchmarking.
    '''
    configs = sorted(CONFIGS.keys(), key=(lambda c: -get_cost(CONFIGS[c])))
    parallel_map(generate_one, configs)


def generate_one(name):
    dataset = loom.store.get_dataset(name)
    if not all(os.path.exists(f) for f in dataset.itervalues()):
        print 'generating', name
        config = CONFIGS[name]
        chunk_size = max(10, (config['row_count'] + 7) / 8)
        mkdir_p(dataset['root'])
        loom.generate.generate(
            init_out=dataset['init'],
            rows_out=dataset['rows'],
            model_out=dataset['model'],
            groups_out=dataset['groups'],
            **config)
        loom.format.make_fake_encoding(
            model_in=dataset['model'],
            rows_in=dataset['rows'],
            schema_out=dataset['schema'],
            encoding_out=dataset['encoding'])
        loom.format.export_rows(
            encoding_in=dataset['encoding'],
            rows_in=dataset['rows'],
            rows_csv_out=dataset['rows_csv'],
            chunk_size=chunk_size)
        loom.generate.generate_init(
            encoding_in=dataset['encoding'],
            model_out=dataset['init'])
        loom.runner.shuffle(
            rows_in=dataset['rows'],
            rows_out=dataset['shuffled'])


@parsable.command
def load(name, schema, rows_csv):
    '''
    Load a csv dataset for testing and benchmarking.
    '''
    assert os.path.exists(schema)
    assert schema.endswith('.json')
    assert os.path.exists(rows_csv)
    if os.path.isfile(rows_csv):
        assert rows_csv.endswith('.csv') or rows_csv.endswith('.csv.gz')
    else:
        assert os.path.isdir(rows_csv)
    dataset = loom.store.get_dataset(name)
    assert not os.path.exists(dataset['root']), 'dataset already loaded'
    os.makedirs(dataset['root'])
    json_dump(json_load(schema), dataset['schema'])
    if os.path.isdir(rows_csv):
        os.symlink(rows_csv, dataset['rows_csv'])
    else:
        os.makedirs(dataset['rows_csv'])
        os.symlink(
            rows_csv,
            os.path.join(dataset['rows_csv'], os.path.basename(rows_csv)))


@parsable.command
def clean(name=None):
    '''
    Clean out one or all datasets.
    '''
    if name is None:
        loom.store.clean_datasets()
    else:
        loom.store.clean_dataset(name)


if __name__ == '__main__':
    parsable.dispatch()
