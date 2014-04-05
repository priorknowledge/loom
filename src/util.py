import os
import traceback
import multiprocessing


def print_trace((fun, arg)):
    try:
        return fun(arg)
    except Exception, e:
        print e
        traceback.print_exc()
        raise


POOL = None


def parallel_map(fun, args):
    THREADS = int(os.environ.get('THREADS', multiprocessing.cpu_count()))
    print 'Running %s in %d threads' % (fun.__name__, THREADS)
    if THREADS == 1:
        return map(fun, args)
    else:
        global POOL
        if POOL is None:
            POOL = multiprocessing.Pool(THREADS)
        fun_args = [(fun, arg) for arg in args]
        return POOL.map(print_trace, fun_args, chunksize=1)
