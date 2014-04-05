import os
import parsable
import subprocess
parsable = parsable.Parsable()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(ROOT, 'build', 'src')
LOOM = os.path.join(BIN, 'loom')


@parsable.command
def run(model_in, values_in, groups_out, safe=True):
    '''
    Run loom.
    '''
    command = [LOOM, model_in, values_in, groups_out]
    print ' \\\n'.join(command)
    if safe:
        subprocess.check_call(command)
    else:
        result = os.system(' '.join(command))
        assert result == 0, 'commaned failed'


if __name__ == '__main__':
    parsable.dispatch()
