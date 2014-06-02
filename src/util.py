import os
import shutil
import traceback
import multiprocessing


def mkdir_p(dirname):
    'like mkdir -p'
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def rm_rf(dirname):
    'like rm -rf'
    if os.path.exists(dirname):
        shutil.rmtree(dirname)


def print_trace((fun, arg)):
    try:
        return fun(arg)
    except Exception, e:
        print e
        traceback.print_exc()
        raise


POOL = None


def parallel_map(fun, args):
    if not isinstance(args, list):
        args = list(args)
    THREADS = int(os.environ.get('THREADS', multiprocessing.cpu_count()))
    if THREADS == 1 or len(args) < 2:
        print 'Running {} in this thread'.format(fun.__name__)
        return map(fun, args)
    else:
        print 'Running {} in {:d} threads'.format(fun.__name__, THREADS)
        global POOL
        if POOL is None:
            POOL = multiprocessing.Pool(THREADS)
        fun_args = [(fun, arg) for arg in args]
        return POOL.map(print_trace, fun_args, chunksize=1)
