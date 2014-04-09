import os
import subprocess
import signal
import parsable
import resource
parsable = parsable.Parsable()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOOM = {
    'release': os.path.join(ROOT, 'build', 'release', 'src', 'loom'),
    'debug': os.path.join(ROOT, 'build', 'debug', 'src', 'loom'),
}


@parsable.command
def run(
        model_in,
        groups_in,
        rows_in,
        groups_out,
        extra_passes=0.0,
        debug=False):
    '''
    Run loom.
    '''
    if groups_in is None:
        groups_in = '--empty'
    if not os.path.exists(groups_out):
        os.makedirs(groups_out)
    loom = LOOM['debug'] if debug else LOOM['release']
    command = [loom, model_in, groups_in, rows_in, groups_out, extra_passes]
    command = map(str, command)
    print ' \\\n'.join(command)
    resource.setrlimit(resource.RLIMIT_CORE, (-1, -1))
    proc = subprocess.Popen(command)
    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.send_signal(signal.SIGQUIT)


if __name__ == '__main__':
    parsable.dispatch()
