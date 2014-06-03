import os
import shutil
import traceback
import multiprocessing
from google.protobuf.descriptor import FieldDescriptor


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


def protobuf_to_dict(message):
    assert message.IsInitialized()
    raw = {}
    for field in message.DESCRIPTOR.fields:
        value = getattr(message, field.name)
        if field.label == FieldDescriptor.LABEL_REPEATED:
            if field.type == FieldDescriptor.TYPE_MESSAGE:
                value = map(protobuf_to_dict, value)
            else:
                value = list(value)
            if len(value) == 0:
                value = None
        else:
            if field.type == FieldDescriptor.TYPE_MESSAGE:
                if value.IsInitialized():
                    value = protobuf_to_dict(value)
                else:
                    value = None
        if value is not None:
            raw[field.name] = value
    return raw


def dict_to_protobuf(raw, message):
    assert isinstance(raw, dict)
    for key, raw_value in raw.iteritems():
        if isinstance(raw_value, dict):
            value = getattr(message, key)
            dict_to_protobuf(raw_value, value)
        elif isinstance(raw_value, list):
            value = getattr(message, key)
            list_to_protobuf(raw_value, value)
        else:
            setattr(message, key, raw_value)


def list_to_protobuf(raw, message):
    assert isinstance(raw, list)
    if raw:
        if isinstance(raw[0], dict):
            for value in raw:
                dict_to_protobuf(value, message.add())
        elif isinstance(raw[0], list):
            for value in raw:
                list_to_protobuf(value, message.add())
        else:
            message[:] = raw
