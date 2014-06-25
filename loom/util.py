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
import shutil
import tempfile
import traceback
import contextlib
import multiprocessing
from google.protobuf.descriptor import FieldDescriptor
from google.protobuf.reflection import GeneratedProtocolMessageType
from distributions.io.stream import protobuf_stream_write, protobuf_stream_read

THREADS = int(os.environ.get('THREADS', multiprocessing.cpu_count()))


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


def protobuf_server(fun, Request, Response):
    assert isinstance(Request, GeneratedProtocolMessageType), Request
    assert isinstance(Response, GeneratedProtocolMessageType), Response

    class Server(object):
        def __init__(self, *args, **kwargs):
            kwargs['block'] = False
            self.proc = fun(*args, **kwargs)

        def call_string(self, request_string):
            protobuf_stream_write(request_string, self.proc.stdin)
            return protobuf_stream_read(self.proc.stdout)

        def call_protobuf(self, request):
            assert isinstance(request, Request)
            request_string = request.SerializeToString()
            response_string = self.call_string(request_string)
            response = Response()
            response.ParseFromString(response_string)
            return response

        def call_dict(self, request_dict):
            request = Request()
            dict_to_protobuf(request_dict, request)
            response = self.call_protobuf(request)
            return protobuf_to_dict(response)

        __call__ = call_protobuf

        def close(self):
            self.proc.stdin.close()
            self.proc.wait()

        def __enter__(self):
            return self

        def __exit__(self, type, value, traceback):
            self.close()

    Server.__name__ = fun.__name__.capitalize() + 'Server'
    return Server


def protobuf_serving(Request, Response):

    def decorator(fun):
        fun.serve = protobuf_server(fun, Request, Response)
        return fun

    return decorator
