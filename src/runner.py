import os
import parsable
import subprocess
parsable = parsable.Parsable()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOOM = {
    'release': os.path.join(ROOT, 'build', 'release', 'src', 'loom'),
    'debug': os.path.join(ROOT, 'build', 'debug', 'src', 'loom'),
}


@parsable.command
def run(model_in, values_in, groups_out, debug=False, safe=True):
    '''
    Run loom.
    '''
    loom = LOOM['debug'] if debug else LOOM['release']
    command = [loom, model_in, values_in, groups_out]
    print ' \\\n'.join(command)
    if safe:
        subprocess.check_call(command)
    else:
        result = os.system(' '.join(command))
        assert result == 0, 'commaned failed'


if __name__ == '__main__':
    parsable.dispatch()
