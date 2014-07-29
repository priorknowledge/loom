import sys
import subprocess


def test_dataflow():
    subprocess.check_call([
        sys.executable, '-m', 'loom.documented', 'make-dataflow',
    ])
