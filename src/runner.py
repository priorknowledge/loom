import os
import parsable
import subprocess
parsable = parsable.Parsable()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(ROOT, 'build', 'src')
LOOM = os.path.join(BIN, 'loom')


@parsable.command
def run(model_in, values_in, groups_out):
    '''
    Run loom.
    '''
    subprocess.check_call([LOOM, model_in, values_in, groups_out])


if __name__ == '__main__':
    parsable.dispatch()
