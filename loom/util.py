# Copyright (c) 2014, Salesforce.com, Inc.  All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# - Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
# - Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
# - Neither the name of Salesforce.com nor the names of its contributors
#   may be used to endorse or promote products derived from this
#   software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
# OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
# TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
# USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
import sys
import shutil
import tempfile
import traceback
import contextlib
import multiprocessing
import simplejson as json
from google.protobuf.descriptor import FieldDescriptor
from distributions.io.stream import (
    open_compressed,
    json_load,
    protobuf_stream_load,
)
import loom.schema_pb2
import parsable
parsable = parsable.Parsable()

THREADS = int(os.environ.get('LOOM_THREADS', multiprocessing.cpu_count()))
VERBOSITY = int(os.environ.get('LOOM_VERBOSITY', 1))


class LoomError(Exception):
    pass


class KnownBug(LoomError):
    pass


def fixme(name, message):
    message = 'FIXME({}) {}'.format(name, message)
    if 'nose' in sys.modules:
        import nose
        return nose.SkipTest(message)
    else:
        return KnownBug(message)


def LOG(message):
    if VERBOSITY:
        sys.stdout.write('{}\n'.format(message))
        sys.stdout.flush()


@contextlib.contextmanager
def chdir(wd):
    oldwd = os.getcwd()
    try:
        os.chdir(wd)
        yield wd
    finally:
        os.chdir(oldwd)


@contextlib.contextmanager
def tempdir(cleanup_on_error=True):
    oldwd = os.getcwd()
    wd = tempfile.mkdtemp()
    try:
        os.chdir(wd)
        yield wd
        cleanup_on_error = True
    finally:
        os.chdir(oldwd)
        if cleanup_on_error:
            shutil.rmtree(wd)


def mkdir_p(dirname):
    'like mkdir -p'
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def rm_rf(path):
    'like rm -rf'
    if os.path.exists(path):
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)


def cp_ns(source, destin):
    'like cp -ns, link destin to source if destin does not exist'
    if not os.path.exists(destin):
        assert os.path.exists(source), source
        dirname = os.path.dirname(destin)
        if dirname and not os.path.exists(dirname):
            os.makedirs(dirname)
        os.symlink(source, destin)


def print_trace((fun, arg)):
    try:
        return fun(arg)
    except Exception, e:
        print e
        traceback.print_exc()
        raise


def parallel_map(fun, args):
    if not isinstance(args, list):
        args = list(args)
    if THREADS == 1 or len(args) < 2:
        print 'Running {} in this thread'.format(fun.__name__)
        return map(fun, args)
    else:
        print 'Running {} in {:d} threads'.format(fun.__name__, THREADS)
        pool = multiprocessing.Pool(THREADS)
        fun_args = [(fun, arg) for arg in args]
        return pool.map(print_trace, fun_args, chunksize=1)


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


GUESS_MESSAGE_TYPE = {
    'rows': 'Row',
    'diffs': 'Row',
    'shuffled': 'Row',
    'tares': 'ProductValue',
    'schema': 'ProductValue',
    'assign': 'Assignment',
    'model': 'CrossCat',
    'init': 'CrossCat',
    'mixture': 'ProductModel.Group',
    'config': 'Config',
    'checkpoint': 'Checkpoint',
    'log': 'LogMessage',
    'infer_log': 'LogMessage',
    'query_log': 'LogMessage',
    'requests': 'Query.Request',
    'responses': 'Query.Response',
}


def get_message(filename, message_type='guess'):
    if message_type == 'guess':
        prefix = os.path.basename(filename).split('.')[0]
        try:
            message_type = GUESS_MESSAGE_TYPE[prefix]
        except KeyError:
            raise LoomError(
                'Cannot guess message type for {}'.format(filename))
    Message = loom.schema_pb2
    for attr in message_type.split('.'):
        Message = getattr(Message, attr)
    return Message()


@parsable.command
def cat(filename, message_type='guess'):
    '''
    Print a text/json/protobuf message from a raw/gz/bz2 file.
    '''
    parts = os.path.basename(filename).split('.')
    if parts[-1] in ['gz', 'bz2']:
        parts.pop()
    protocol = parts[-1]
    if protocol == 'json':
        data = json_load(filename)
        print json.dumps(data, sort_keys=True, indent=4)
    elif protocol == 'pb':
        message = get_message(filename, message_type)
        with open_compressed(filename) as f:
            message.ParseFromString(f.read())
            print message
    elif protocol == 'pbs':
        message = get_message(filename, message_type)
        for string in protobuf_stream_load(filename):
            message.ParseFromString(string)
            print message
    else:
        with open_compressed(filename) as f:
            for line in f:
                print line


if __name__ == '__main__':
    parsable.dispatch()
