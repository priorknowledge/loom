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

BASENAMES = {
    'ingest': {
        'version': 'version.txt',
        'schema': 'schema.json.gz',
        'rows_csv': 'rows_csv',
        'rowids': 'rowids.csv.gz',
        'encoding': 'encoding.json.gz',
        'rows': 'rows.pbs.gz',
        'schema_row': 'schema.pb.gz',
        'tares': 'tares.pbs.gz',
        'diffs': 'diffs.pbs.gz',
    },
    'sample': {
        'config': 'config.pb.gz',
        'init': 'init.pb.gz',
        'shuffled': 'shuffled.pbs.gz',
        'model': 'model.pb.gz',
        'groups': 'groups',
        'assign': 'assign.pbs.gz',
        'infer_log': 'infer_log.pbs',
    },
    'consensus': {
        'config': 'config.pb.gz',
        'model': 'model.pb.gz',
        'groups': 'groups',
        'assign': 'assign.pbs.gz',
    },
    'query': {
        'query_log': 'query_log.pbs',
        'config': 'config.pb.gz',
        'similar_diffs': 'similar_diffs.pbs.gz',
    },
}

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


def get_mixture_path(groups_path, kindid):
    '''
    This must match loom::store::get_mixture_path(-,-) in src/store.hpp
    '''
    assert kindid >= 0
    return os.path.join(groups_path, 'mixture.{:d}.pbs.gz'.format(kindid))


get_mixture_filename = get_mixture_path  # DEPRECATED


def get_sample_path(root, seed):
    '''
    This must match loom::store::get_sample_path(-,-) in src/store.hpp
    '''
    assert seed >= 0
    return os.path.join(root, 'samples', 'sample.{:d}'.format(seed))


def join_paths(*args):
    args, paths = args[:-1], args[-1]
    return {
        key: os.path.join(*(args + (value,)))
        for key, value in paths.iteritems()
    }


def get_paths(root, sample_count=1):
    assert sample_count >= 0 or sample_count is None, sample_count
    if sample_count is None:
        sample_count = len(os.listdir(os.path.join(root, 'samples')))
    if not os.path.isabs(root):
        root = os.path.join(STORE, root)
    paths = {'root': root}
    paths['ingest'] = join_paths(root, 'ingest', BASENAMES['ingest'])
    paths['consensus'] = join_paths(root, 'consensus', BASENAMES['consensus'])
    paths['query'] = join_paths(root, 'query', BASENAMES['query'])
    paths['samples'] = []
    for seed in xrange(sample_count):
        sample_root = get_sample_path(root, seed)
        paths['samples'].append(join_paths(sample_root, BASENAMES['sample']))
    return paths


def iter_paths(name, paths):
    if isinstance(paths, basestring):
        yield name, paths
    elif isinstance(paths, dict):
        for key, value in paths.iteritems():
            for pair in iter_paths('{}.{}'.format(name, key), value):
                yield pair
    else:
        for i, value in enumerate(paths):
            for pair in iter_paths('{}.{}'.format(name, i), value):
                yield pair


def get_path(paths, chain):
    '''
    Inputs:
        paths - the result of a get_paths(...)
        chain - a . delimited string of keys
    Returns:
        the sub-paths rooted at `paths` descended by `chain`

    Example:
        paths = get_paths(name)
        model_name = get_paths(paths, 'samples.0.model')
    '''
    path = paths
    for key in chain.split('.'):
        try:
            key = int(key)
        except ValueError:
            pass
        path = path[key]
    return path


def path_exists(paths, chain):
    path = get_path(paths, chain)
    return os.path.exists(path)


def require(name, requirements):
    if name is None:
        print 'try one of:'
        for name in sorted(os.listdir(STORE)):
            paths = get_paths(name)
            if all(path_exists(paths, r) for r in requirements):
                print '  {}'.format(name)
        sys.exit(1)
    else:
        paths = get_paths(name)
        for r in requirements:
            error = ERRORS.get(r.split('.')[-1], 'First create {}'.format(r))
            assert path_exists(paths, r), error + '\n  (missing {})'.format(r)


def provide(destin_name, source_paths, requirements):
    destin_paths = get_paths(destin_name)
    for chain in requirements:
        source = get_path(source_paths, chain)
        destin = get_path(destin_paths, chain)
        assert os.path.exists(source), source
        if not os.path.exists(destin):
            dirname = os.path.dirname(destin)
            if dirname and not os.path.exists(dirname):
                os.makedirs(dirname)
            os.symlink(source, destin)
