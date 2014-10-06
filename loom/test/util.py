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
import functools
from nose.tools import assert_true
from distributions.io.stream import protobuf_stream_load
from loom.util import csv_reader
from loom.schema_pb2 import Row
import loom.datasets
import loom.store


CLEANUP_ON_ERROR = int(os.environ.get('CLEANUP_ON_ERROR', 1))


def assert_found(*filenames):
    for name in filenames:
        assert_true(os.path.exists(name), 'missing file: {}'.format(name))


def for_each_dataset(fun):
    @functools.wraps(fun)
    def test_one(dataset):
        paths = loom.store.get_paths(dataset, sample_count=2)
        for key, path in loom.store.iter_paths(dataset, paths):
            if not os.path.exists(path):
                raise ValueError(
                    'missing {} at {},\n  first `python -m loom.datasets test`'
                    .format(key, path))
        kwargs = {'name': dataset}
        kwargs.update(paths['query'])
        kwargs.update(paths['consensus'])
        kwargs.update(paths['samples'][0])
        kwargs.update(paths['ingest'])
        kwargs.update(paths)
        fun(**kwargs)

    @functools.wraps(fun)
    def test_all():
        for dataset in loom.datasets.TEST_CONFIGS:
            yield test_one, dataset

    return test_all


def load_rows(filename):
    rows = []
    for string in protobuf_stream_load(filename):
        row = Row()
        row.ParseFromString(string)
        rows.append(row)
    return rows


def load_rows_raw(filename):
    return list(protobuf_stream_load(filename))


def load_rows_csv(dirname):
    filenames = os.listdir(dirname)
    rows = [None]
    for filename in filenames:
        filename = os.path.join(dirname, filename)
        with csv_reader(filename) as reader:
            header = reader.next()
            rows += reader
    rows[0] = header
    return rows
