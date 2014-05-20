import os
import subprocess
import parsable
from loom.config import DEFAULTS
parsable = parsable.Parsable()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = {
    'release': os.path.join(ROOT, 'build', 'release', 'src'),
    'debug': os.path.join(ROOT, 'build', 'debug', 'src'),
}
PROFILERS = {
    None: [],
    'time': ['/usr/bin/time', '--verbose'],
    'valgrind': ['valgrind', '--leak-check=full', '--track-origins=yes'],
    'cachegrind': ['valgrind', '--tool=cachegrind'],
    'callgrind': [
        'valgrind',
        '--tool=callgrind',
        '--callgrind-out-file=callgrind.out',
    ],
}


def check_call(command, debug, profile):
    build_type = 'debug' if debug else 'release'
    bin_ = os.path.join(BIN[build_type], command[0])
    args = map(str, command[1:])
    command = PROFILERS[profile] + [bin_] + args
    if profile:
        retcode = subprocess.Popen(command).wait()
        print 'Program returned {}'.format(retcode)
    else:
        if debug:
            print ' \\\n  '.join(command)
        subprocess.check_call(command)


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
def shuffle(
        rows_in='-',
        rows_out='-',
        seed=DEFAULTS['seed'],
        debug=False,
        profile=None):
    '''
    Shuffle dataset for inference.
    '''
    command = ['shuffle', rows_in, rows_out, seed]
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
        checkpoint_in=None,
        model_out=None,
        groups_out=None,
        assign_out=None,
        checkpoint_out=None,
        log_out=None,
        debug=False,
        profile=None):
    '''
    Run inference.
    '''
    groups_in = optional_file(groups_in)
    assign_in = optional_file(assign_in)
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
        config_in, rows_in, model_in, groups_in, assign_in, checkpoint_in,
        model_out, groups_out, assign_out, checkpoint_out, log_out,
    ]
    assert_found(
        config_in, rows_in, model_in, groups_in, assign_in, checkpoint_in)
    check_call(command, debug, profile)
    assert_found(model_out, groups_out, assign_out, checkpoint_out, log_out)


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


@parsable.command
def predict(
        config_in,
        model_in,
        groups_in,
        queries_in='-',
        results_out='-',
        debug=False,
        profile=None):
    '''
    Run predictions server.
    '''
    command = [
        'predict',
        config_in,
        model_in,
        groups_in,
        queries_in,
        results_out,
    ]
    assert_found(config_in, model_in, groups_in, queries_in)
    check_call(command, debug, profile)
    assert_found(results_out)


if __name__ == '__main__':
    parsable.dispatch()
