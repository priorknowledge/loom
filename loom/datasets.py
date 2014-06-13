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
