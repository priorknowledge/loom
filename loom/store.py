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

if 'LOOM_STORE' in os.environ:
    STORE = os.environ['LOOM_STORE']
else:
    STORE = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'data')

MAX_SEED = 999999

DATASET_PATHS = {
    'version': 'version.txt',
    'schema': 'schema.json.gz',
    'rows_csv': 'rows_csv',
    'encoding': 'encoding.json.gz',
    'rows': 'rows.pbs.gz',
    'schema_row': 'schema.pb.gz',
    'tares': 'tares.pbs.gz',
    'diffs': 'diffs.pbs.gz',
    'samples': 'samples',
    'consensus': 'consensus',
}

SAMPLE_PATHS = {
    'config': 'config.pb.gz',
    'init': 'init.pb.gz',
    'shuffled': 'shuffled.pbs.gz',
    'model': 'model.pb.gz',
    'groups': 'groups',
    'assign': 'assign.pbs.gz',
    'infer_log': 'infer_log.pbs',
}

CONSENSUS_PATHS = {
    'config': 'config.pb.gz',
    'model': 'model.pb.gz',
    'groups': 'groups',
    'assign': 'assign.pbs.gz',
}


def get_mixture_filename(dirname, kindid):
    '''
    This must match get_mixture_filename(-,-) in src/cross_cat.cc
    '''
    return os.path.join(dirname, 'mixture.{:06d}.pbs.gz'.format(kindid))


def get_sample_dirname(dirname, seed):
    return os.path.join(dirname, 'sample.{:06d}'.format(seed))


def get_dataset_paths(name, operation=None):
    root = name if operation is None else os.path.join(name, operation)
    if not os.path.isabs(root):
        root = os.path.join(STORE, root)
    paths = {'root': root}
    for name, path in DATASET_PATHS.iteritems():
        paths[name] = os.path.join(root, path)
    return paths


def get_paths(name, operation=None, seed=0):
    assert seed < MAX_SEED, seed
    paths = get_dataset_paths(name, operation)
    sample = get_sample_dirname(paths['samples'], int(seed))
    for name, path in SAMPLE_PATHS.iteritems():
        paths[name] = os.path.join(sample, path)
    return paths


def get_consensus(name, operation=None):
    paths = get_dataset_paths(name, operation)
    consensus = paths['consensus']
    consensus_paths = {}
    for name, path in CONSENSUS_PATHS.iteritems():
        consensus_paths[name] = os.path.join(consensus, path)
    return consensus_paths


def get_samples(name, operation=None, sample_count=None):
    if sample_count is None:
        paths = get_paths(name, operation)
        sample_count = len(os.listdir(paths['samples']))
    samples = [
        get_paths(name, operation, seed=seed)
        for seed in xrange(int(sample_count))
    ]
    return samples


ERRORS = {
    'schema': 'First load dataset',
    'schema_row': 'First generate or ingest dataset',
    'config': 'First load or ingest dataset',
    'encoding': 'First load or ingest dataset',
    'rows': 'First generate or ingest dataset',
    'tares': 'First tare dataset',
    'diffs': 'First sparsify dataset',
    'init': 'First init',
    'shuffled': 'First shuffle',
}


def require(name, *requirements):
    if name is None:
        print 'try one of:'
        for name in sorted(os.listdir(STORE)):
            dataset = get_paths(name, 'data')
            if all(os.path.exists(dataset[r]) for r in requirements):
                print '  {}'.format(name)
        sys.exit(1)
    else:
        dataset = get_paths(name, 'data')
        for r in requirements:
            error = ERRORS.get(r, 'First create {}'.format(r))
            assert os.path.exists(dataset[r]), error
