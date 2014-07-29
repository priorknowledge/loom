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
import datetime
from distributions.io.stream import protobuf_stream_read
from loom.schema_pb2 import LogMessage
import parsable
parsable = parsable.Parsable()


# FIXME this does not work with .gz files
def protobuf_stream_watch(filename):
    assert os.path.exists(filename)
    proc = subprocess.Popen(
        ['tail', '-f', '-c', '+0', filename],
        stdout=subprocess.PIPE)
    while True:
        yield protobuf_stream_read(proc.stdout)


def print_page(message):
    print '--------------------------------'
    print message,


def print_line(message):
    sys.stdout.write('\033[s{}\033[u'.format(message))
    sys.stdout.flush()


def usec_to_datetime(epoch_usec):
    epoch = epoch_usec / 1000000
    delta = epoch_usec % 1000000
    timestamp = datetime.datetime.fromtimestamp(epoch)
    timestamp_delta = datetime.timedelta(microseconds=delta)
    return timestamp + timestamp_delta


def pretty_timedelta(delta):
    seconds = int(delta.total_seconds())
    hours = seconds / 3600
    minutes = (seconds % 3600) / 60
    seconds = seconds % 60
    return '{:d}:{:02d}:{:02d}'.format(hours, minutes, seconds)


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
def partial(log_file):
    '''
    Print partial log messages as they are written.
    '''
    start_time = None
    message = LogMessage()
    for string in protobuf_stream_watch(log_file):
        message.ParseFromString(string)
        time = usec_to_datetime(message.timestamp_usec)
        if start_time is None:
            start_time = time
        summary = message.args.summary
        counts = zip(summary.feature_counts, summary.category_counts)
        counts.sort(reverse=True)
        feature_counts = [str(f) for f, o in counts]
        category_counts = [str(o) for f, o in counts]
        part = '\n'.join([
            'time: {}'.format(pretty_timedelta(time - start_time)),
            'iter: {}'.format(message.args.iter),
            'assigned_object_count: {}'.format(
                message.args.scores.assigned_object_count),
            'kind_count: {}'.format(len(feature_counts)),
            'feature_counts: {}'.format(' '.join(feature_counts)),
            'category_counts: {}'.format(' '.join(category_counts)),
            'kernels:\n{}'.format(message.args.kernel_status),
            'rusage:\n{}'.format(message.rusage),
        ])
        print_page(part)


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
