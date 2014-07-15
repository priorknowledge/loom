import os
import loom.util
from loom.test.util import for_each_dataset


@for_each_dataset
def test_cat(**files):
    _test_cat(files)


def _test_cat(files):
    for name, filename in files.iteritems():
        if name not in ['name', 'root', 'groups']:
            if os.path.isdir(filename):
                for f in os.listdir(filename):
                    _test_cat({name: os.path.join(filename, f)})
            else:
                print '==== {} ===='.format(name)
                loom.util.cat(filename)
