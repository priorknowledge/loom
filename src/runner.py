import os
import subprocess
import parsable
parsable = parsable.Parsable()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = {
    'release': os.path.join(ROOT, 'build', 'release', 'src'),
    'debug': os.path.join(ROOT, 'build', 'debug', 'src'),
}


@parsable.command
def infer(
        model_in,
        groups_in=None,
        assign_in=None,
        rows_in='-',
        groups_out=None,
        assign_out=None,
        extra_passes=0.0,
        debug=False):
    '''
    Run loom.
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
    ]
    command = map(str, command)
    print ' \\\n  '.join(command)
    subprocess.check_call(command)


if __name__ == '__main__':
    parsable.dispatch()
