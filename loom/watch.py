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
import subprocess
from distributions.io.stream import protobuf_stream_read
from loom.schema_pb2 import LogMessage
import parsable
parsable = parsable.Parsable()


def protobuf_stream_watch(filename):
    assert os.path.exists(filename)
    proc = subprocess.Popen(
        ['tail', '-f', '-c', '+0', filename],
        stdout=subprocess.PIPE)
    while True:
        yield protobuf_stream_read(proc.stdout)


def print_page(message):
    sys.stdout.write('\033[2J{}'.format(message))
    sys.stdout.flush()


def print_line(message):
    sys.stdout.write('\033[s{}\033[u'.format(message))
    sys.stdout.flush()


@parsable.command
def full(log_file):
    '''
    Print brief log messages as they are written.
    '''
    message = LogMessage()
    for string in protobuf_stream_watch(log_file):
        message.ParseFromString(string)
        print_page(message)


@parsable.command
def brief(log_file, delay_sec=3.0):
    '''
    Print brief log messages as they are written.
    '''
    print 'iter\tassigned_object_count'
    message = LogMessage()
    for string in protobuf_stream_watch(log_file):
        message.ParseFromString(string)
        print_line('{}\t{}'.format(
            message.args.iter,
            message.args.scores.assigned_object_count))


if __name__ == '__main__':
    parsable.dispatch()
