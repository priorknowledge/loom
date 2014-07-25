import os
import shutil
import loom.store


def copy_all(src, dst):
    if os.path.isdir(src):
        shutil.copytree(src, dst)
    else:
        shutil.copy(src, dst)


def get_consensus(name, debug=False):
    sample_paths = loom.store.get_samples(name)[0]
    consensus_paths = loom.store.get_consensus(name)
    for name, path in consensus_paths.iteritems():
        copy_all(sample_paths[name], path)
