import os
import subprocess
import parsable
parsable = parsable.Parsable()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = {
    'release': os.path.join(ROOT, 'build', 'release', 'src'),
    'debug': os.path.join(ROOT, 'build', 'debug', 'src'),
}


def assert_found(*filenames):
    for filename in filenames:
        if filename not in ['-', '-.gz', '--none', None]:
            if not os.path.exists(filename):
                raise IOError('File not found: {}'.format(filename))


@parsable.command
def infer(
        model_in,
        groups_in=None,
        assign_in=None,
        rows_in='-',
        groups_out=None,
        assign_out=None,
        extra_passes=0.0,
        kind_count=0,
        debug=False):
    '''
    Run inference.
    '''
    if groups_in is None:
        groups_in = '--none'
    if assign_in is None:
        assign_in = '--none'
    if assign_out is None:
        assign_out = '--none'
    if not os.path.exists(groups_out):
        os.makedirs(groups_out)
    build_type = 'debug' if debug else 'release'
    command = [
        os.path.join(BIN[build_type], 'infer'),
        model_in,
        groups_in,
        assign_in,
        rows_in,
        groups_out,
        assign_out,
        extra_passes,
        kind_count,
    ]
    command = map(str, command)
    assert_found(model_in, groups_in, assign_in, rows_in)
    print ' \\\n  '.join(command)
    subprocess.check_call(command)
    assert_found(groups_out, assign_out)


@parsable.command
def predict(model_in, groups_in, queries_in='-', results_out='-', debug=False):
    '''
    Run predictions server.
    '''
    build_type = 'debug' if debug else 'release'
    command = [
        os.path.join(BIN[build_type], 'predict'),
        model_in,
        groups_in,
        queries_in,
        results_out,
    ]
    command = map(str, command)
    assert_found(model_in, groups_in, queries_in)
    print ' \\\n  '.join(command)
    subprocess.check_call(command)
    assert_found(results_out)


if __name__ == '__main__':
    parsable.dispatch()
