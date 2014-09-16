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
import loom.documented
from loom.config import DEFAULTS
import loom.config
import loom.store
parsable = parsable.Parsable()

PROFILERS = {
    'none': [],
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


def which(binary):
    path = subprocess.check_output('which {}'.format(binary), shell=True)
    return path.strip()


def popen_piped(command, debug, profile):
    profile = str(profile).lower()
    bin_pattern = 'loom_{}_debug' if debug else 'loom_{}'
    bin_ = [which(bin_pattern.format(command[0]))]
    args = map(str, command[1:])
    command = PROFILERS[profile] + bin_ + args
    return subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        bufsize=-1)


def check_call(command, debug, profile, **kwargs):
    profile = str(profile).lower()
    if command[0] == 'python':
        bin_ = [PYTHON] if debug else [PYTHON, '-O']
    else:
        bin_pattern = 'loom_{}_debug' if debug else 'loom_{}'
        bin_ = [which(bin_pattern.format(command[0]))]
    args = map(str, command[1:])
    command = PROFILERS[profile] + bin_ + args
    if profile != 'none':
        retcode = subprocess.Popen(command, **kwargs).wait()
        print 'Program returned {}'.format(retcode)
    else:
        if debug:
            print ' \\\n  '.join(command)
        subprocess.check_call(command, **kwargs)


def check_call_files(command, debug, profile, infiles=[], outfiles=[]):
    assert_found(infiles)
    make_dirs_for(outfiles)
    check_call(command, debug, profile)
    assert_found(outfiles)


FAKE_FILES = frozenset(['-', '-.gz', '--none', None])
DIRNAMES = set(['ingest', 'infer', 'groups'])


def make_dirs_for(filenames):
    for filename in filenames:
        if filename not in FAKE_FILES:
            if os.path.basename(filename) in DIRNAMES:
                dirname = filename
            else:
                dirname = os.path.dirname(filename)
            if dirname and not os.path.exists(dirname):
                os.makedirs(dirname)


def assert_found(filenames):
    for filename in filenames:
        if filename not in FAKE_FILES:
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
@loom.documented.transform(
    inputs=['ingest.schema_row', 'ingest.rows'],
    outputs=['ingest.tares'])
def tare(
        schema_row_in,
        rows_in,
        tares_out,
        debug=False,
        profile=None):
    '''
    Find tare rows for a datset, i.e., rows of per-column most-likely values.
    '''
    check_call_files(
        command=['tare', schema_row_in, rows_in, tares_out],
        debug=debug,
        profile=profile,
        infiles=[schema_row_in, rows_in],
        outfiles=[tares_out])


@parsable.command
@loom.documented.transform(
    inputs=['ingest.schema_row', 'ingest.tares', 'ingest.rows'],
    outputs=['ingest.diffs'])
def sparsify(
        schema_row_in,
        tares_in,
        rows_in='-',
        rows_out='-',
        debug=False,
        profile=None):
    '''
    Sparsify dataset WRT tare rows.
    '''
    check_call_files(
        command=['sparsify', schema_row_in, tares_in, rows_in, rows_out],
        debug=debug,
        profile=profile,
        infiles=[schema_row_in, tares_in, rows_in],
        outfiles=[rows_out])


@parsable.command
@loom.documented.transform(
    inputs=['ingest.diffs', 'seed'],
    outputs=['samples.0.shuffled'])
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
    check_call_files(
        command=['shuffle', rows_in, rows_out, seed, target_mem_bytes],
        debug=debug,
        profile=profile,
        infiles=[rows_in],
        outfiles=[rows_out])


@parsable.command
@loom.documented.transform(
    inputs=[
        'samples.0.config',
        'samples.0.shuffled',
        'ingest.tares',
        'samples.0.init'],
    outputs=[
        'samples.0.model',
        'samples.0.groups',
        'samples.0.assign',
        'samples.0.infer_log'])
def infer(
        config_in,
        rows_in,
        model_in,
        tares_in=None,
        groups_in=None,
        assign_in=None,
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
    tares_in = optional_file(tares_in)
    checkpoint_in = optional_file(checkpoint_in)
    model_out = optional_file(model_out)
    groups_out = optional_file(groups_out)
    assign_out = optional_file(assign_out)
    checkpoint_out = optional_file(checkpoint_out)
    log_out = optional_file(log_out)

    check_call_files(
        command=[
            'infer',
            config_in, rows_in, tares_in,
            model_in, groups_in, assign_in, checkpoint_in,
            model_out, groups_out, assign_out, checkpoint_out, log_out,
        ],
        debug=debug,
        profile=profile,
        infiles=[
            config_in, rows_in, tares_in,
            model_in, groups_in, assign_in, checkpoint_in,
        ],
        outfiles=[model_out, groups_out, assign_out, checkpoint_out, log_out])


@parsable.command
@loom.documented.transform(
    inputs=['samples.0.config', 'samples.0.model'],
    outputs=[
        'ingest.rows',
        'samples.0.model',
        'samples.0.groups',
        'samples.0.assign'],
    role='test')
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

    check_call_files(
        command=[
            'generate',
            config_in, model_in,
            rows_out, model_out, groups_out, assign_out,
        ],
        debug=debug,
        profile=profile,
        infiles=[config_in, model_in],
        outfiles=[rows_out, model_out, groups_out, assign_out])


@parsable.command
def mix(config_in,
        rows_in,
        model_in,
        groups_in,
        assign_in,
        model_out,
        groups_out,
        assign_out,
        debug=False,
        profile=None):
    '''
    Generate additional samples of a dataset.
    '''
    check_call_files(
        command=[
            'mix',
            config_in, rows_in, model_in, groups_in, assign_in,
            model_out, groups_out, assign_out,
        ],
        debug=debug,
        profile=profile,
        infiles=[config_in, rows_in, model_in, groups_in, assign_in],
        outfiles=[model_out, groups_out, assign_out])


@parsable.command
def posterior_enum(
        config_in,
        model_in,
        rows_in,
        samples_out,
        tares_in=None,
        groups_in=None,
        assign_in=None,
        debug=False,
        profile=None):
    '''
    Generate samples for posterior enumeration tests.
    '''
    tares_in = optional_file(tares_in)
    groups_in = optional_file(groups_in)
    assign_in = optional_file(assign_in)

    check_call_files(
        command=[
            'posterior_enum',
            config_in, rows_in, tares_in, model_in, groups_in, assign_in,
            samples_out,
        ],
        debug=debug,
        profile=profile,
        infiles=[config_in, rows_in, tares_in, model_in, groups_in, assign_in],
        outfiles=[samples_out])


@parsable.command
def query(
        root_in,
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
    config_in = loom.store.get_paths(root_in)['query']['config']
    command = [
        'query',
        root_in,
        requests_in,
        responses_out,
        config_in,
        log_out]
    infiles = [root_in, requests_in]
    if block:
        check_call_files(
            command=command,
            debug=debug,
            profile=profile,
            infiles=infiles,
            outfiles=[responses_out, log_out])
    else:
        assert requests_in == '-', 'cannot pipe requests'
        assert responses_out == '-', 'cannot pipe responses'
        assert_found(infiles)
        return popen_piped(command, debug, profile)
