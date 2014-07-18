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


def get_paths(*parts):
    root = os.path.join(STORE, *map(str, parts))
    return {
        'root': root,
        'version': os.path.join(root, 'version.txt'),
        'config': os.path.join(root, 'config.pb.gz'),
        'rows': os.path.join(root, 'rows.pbs.gz'),
        'schema_row': os.path.join(root, 'schema.pb.gz'),
        'tares': os.path.join(root, 'tares.pbs.gz'),
        'diffs': os.path.join(root, 'diffs.pbs.gz'),
        'init': os.path.join(root, 'init.pb.gz'),
        'shuffled': os.path.join(root, 'shuffled.pbs.gz'),
        'model': os.path.join(root, 'model.pb.gz'),
        'groups': os.path.join(root, 'groups'),
        'assign': os.path.join(root, 'assign.pbs.gz'),
        'rows_csv': os.path.join(root, 'rows_csv'),
        'schema': os.path.join(root, 'schema.json.gz'),
        'encoding': os.path.join(root, 'encoding.json.gz'),
        'infer_log': os.path.join(root, 'infer_log.pbs'),
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
