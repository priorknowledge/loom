import os
import shutil
import loom.store
from loom.util import mkdir_p


def copy_all(src, dst):
    if os.path.exists(dst):
        return
    elif os.path.isdir(src):
        shutil.copytree(src, dst)
    else:
        shutil.copy(src, dst)


def make_fake_consensus(name, debug=False):
    paths = loom.store.get_consensus(name)
    sample_paths = loom.store.get_paths(name, seed=0)
    mkdir_p(paths['consensus'])
    for name in loom.store.CONSENSUS_PATHS:
        if name != 'config':
            copy_all(sample_paths[name], paths[name])


def get_consensus(name, debug=False):
    make_fake_consensus(name, debug)
