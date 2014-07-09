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
import subprocess
import parsable
from loom.config import DEFAULTS
from loom.schema_pb2 import Query
from loom.util import protobuf_serving
parsable = parsable.Parsable()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = {
    'release': os.path.join(ROOT, 'build', 'release', 'src'),
    'debug': os.path.join(ROOT, 'build', 'debug', 'src'),
}
PROFILERS = {
    'None': [],
    'time': ['/usr/bin/time', '--verbose'],
    'valgrind': ['valgrind', '--leak-check=full', '--track-origins=yes'],
    'cachegrind': ['valgrind', '--tool=cachegrind'],
    'callgrind': [
        'valgrind',
        '--tool=callgrind',
        '--callgrind-out-file=callgrind.out',
    ],
    'helgrind': ['valgrind', '--tool=helgrind', '--read-var-info=yes'],
}
PYTHON = sys.executable


def popen_piped(command, debug):
    build_type = 'debug' if debug else 'release'
    bin_ = os.path.join(BIN[build_type], 'loom_' + command[0])
    args = map(str, command[1:])
    command = [bin_] + args
    return subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE)


def check_call(command, debug, profile, **kwargs):
    profile = str(profile)
    if command[0] == 'python':
        bin_ = [PYTHON] if debug else [PYTHON, '-O']
    else:
        build_type = 'debug' if debug else 'release'
        bin_ = [os.path.join(BIN[build_type], 'loom_' + command[0])]
    args = map(str, command[1:])
    command = PROFILERS[profile] + bin_ + args
    if profile != 'None':
        retcode = subprocess.Popen(command, **kwargs).wait()
        print 'Program returned {}'.format(retcode)
    else:
        if debug:
            print ' \\\n  '.join(command)
        subprocess.check_call(command, **kwargs)


def assert_found(*filenames):
    for filename in filenames:
        if filename not in ['-', '-.gz', '--none', None]:
            if not os.path.exists(filename):
                raise IOError('File not found: {}'.format(filename))


def optional_file(filename):
    return '--none' if filename is None else filename


@parsable.command
def profilers():
    '''
    Print available profilers.
    '''
    for key, value in sorted(PROFILERS.iteritems()):
        print '  {} = {}'.format(key, ' '.join(value))


@parsable.command
def tare(
        schema_row_in,
        rows_in,
        tare_out,
        debug=False,
        profile=None):
    '''
    Find a tare row for a datset, i.e., a row of per-column most-likely values.
    '''
    command = ['tare', schema_row_in, rows_in, tare_out]
    assert_found(schema_row_in, rows_in)
    check_call(command, debug, profile)
    assert_found(tare_out)


@parsable.command
def sparsify(
        config_in,
        schema_row_in,
        tare_in,
        rows_in='-',
        rows_out='-',
        debug=False,
        profile=None):
    '''
    Sparsify dataset WRT a tare row.
    '''
    command = [
        'sparsify',
        config_in, schema_row_in, tare_in, rows_in,
        rows_out,
    ]
    assert_found(config_in, schema_row_in, tare_in, rows_in)
    check_call(command, debug, profile)
    assert_found(rows_out)


@parsable.command
def shuffle(
        rows_in,
        rows_out='-',
        seed=DEFAULTS['seed'],
        target_mem_bytes=DEFAULTS['target_mem_bytes'],
        debug=False,
        profile=None):
    '''
    Shuffle a dataset for inference.
    '''
    assert rows_in != rows_out, 'cannot shuffle rows in-place'
    command = ['shuffle', rows_in, rows_out, seed, target_mem_bytes]
    assert_found(rows_in)
    check_call(command, debug, profile)
    assert_found(rows_out)


@parsable.command
def infer(
        config_in,
        rows_in,
        model_in,
        groups_in=None,
        assign_in=None,
        tare_in=None,
        checkpoint_in=None,
        model_out=None,
        groups_out=None,
        assign_out=None,
        checkpoint_out=None,
        log_out=None,
        debug=False,
        profile=None):
    '''
    Run inference on a dataset.
    '''
    groups_in = optional_file(groups_in)
    assign_in = optional_file(assign_in)
    tare_in = optional_file(tare_in)
    checkpoint_in = optional_file(checkpoint_in)
    model_out = optional_file(model_out)
    groups_out = optional_file(groups_out)
    assign_out = optional_file(assign_out)
    checkpoint_out = optional_file(checkpoint_out)
    log_out = optional_file(log_out)
    if groups_out != '--none' and not os.path.exists(groups_out):
        os.makedirs(groups_out)
    command = [
        'infer',
        config_in, rows_in, model_in,
        groups_in, assign_in, tare_in, checkpoint_in,
        model_out, groups_out, assign_out, checkpoint_out, log_out,
    ]
    assert_found(
        config_in, rows_in, model_in,
        groups_in, assign_in, tare_in, checkpoint_in)
    check_call(command, debug, profile)
    assert_found(model_out, groups_out, assign_out, checkpoint_out, log_out)


@parsable.command
def generate(
        config_in,
        model_in,
        rows_out,
        model_out=None,
        groups_out=None,
        assign_out=None,
        debug=False,
        profile=None):
    '''
    Generate a synthetic dataset.
    '''
    model_out = optional_file(model_out)
    groups_out = optional_file(groups_out)
    assign_out = optional_file(assign_out)
    if groups_out != '--none' and not os.path.exists(groups_out):
        os.makedirs(groups_out)
    command = [
        'generate',
        config_in, model_in,
        rows_out, model_out, groups_out, assign_out,
    ]
    assert_found(config_in, model_in)
    check_call(command, debug, profile)
    assert_found(rows_out, model_out, groups_out, assign_out)


@parsable.command
def posterior_enum(
        config_in,
        model_in,
        rows_in,
        samples_out,
        groups_in=None,
        assign_in=None,
        debug=False,
        profile=None):
    '''
    Generate samples for posterior enumeration tests.
    '''
    groups_in = optional_file(groups_in)
    assign_in = optional_file(assign_in)
    command = [
        'posterior_enum',
        config_in, model_in, groups_in, assign_in, rows_in,
        samples_out,
    ]
    assert_found(config_in, model_in, groups_in, assign_in, rows_in)
    check_call(command, debug, profile)
    assert_found(samples_out)


@protobuf_serving(Query.Request, Query.Response)
@parsable.command
def query(
        config_in,
        model_in,
        groups_in,
        requests_in='-',
        responses_out='-',
        log_out=None,
        debug=False,
        profile=None,
        block=True):
    '''
    Run query server from a trained model.
    '''
    log_out = optional_file(log_out)
    command = [
        'query',
        config_in, model_in, groups_in, requests_in,
        responses_out, log_out,
    ]
    assert_found(config_in, model_in, groups_in, requests_in)
    if block:
        check_call(command, debug, profile)
        assert_found(responses_out, log_out)
    else:
        assert requests_in == '-', 'cannot pipe requests'
        assert responses_out == '-', 'cannot pipe responses'
        return popen_piped(command, debug)
