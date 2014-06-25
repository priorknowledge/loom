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
from itertools import izip
from distributions.fileutil import tempdir
from nose.tools import assert_equal
import mock
from distributions.tests.util import assert_close
import loom.format
import loom.util
from loom.test.util import for_each_dataset, CLEANUP_ON_ERROR, assert_found
from distributions.io.stream import json_load, protobuf_stream_load


@for_each_dataset
def test_make_fake_encoding(model, rows, **unused):
    with tempdir(cleanup_on_error=CLEANUP_ON_ERROR):
        schema_out = os.path.abspath('schema.json.gz')
        encoding_out = os.path.abspath('encoding.json.gz')
        loom.format.make_fake_encoding(
            model_in=model,
            rows_in=rows,
            schema_out=schema_out,
            encoding_out=encoding_out)
        assert_found(schema_out, encoding_out)


@for_each_dataset
def test_make_encoding(schema, rows_csv, **unused):
    with tempdir(cleanup_on_error=CLEANUP_ON_ERROR):
        encoding = os.path.abspath('encoding.json.gz')
        rows = os.path.abspath('rows.pbs.gz')
        loom.format.make_encoding(
            schema_in=schema,
            rows_in=rows_csv,
            encoding_out=encoding)
        assert_found(encoding)
        loom.format.import_rows(
            encoding_in=encoding,
            rows_in=rows_csv,
            rows_out=rows)
        assert_found(rows)


@for_each_dataset
def test_import_rows(encoding, rows_csv, **unused):
    with tempdir(cleanup_on_error=CLEANUP_ON_ERROR):
        rows = os.path.abspath('rows.pbs.gz')
        loom.format.import_rows(
            encoding_in=encoding,
            rows_in=rows_csv,
            rows_out=rows)
        assert_found(rows)


@for_each_dataset
def test_parallel(schema, rows_csv, **unused):
    with tempdir(cleanup_on_error=CLEANUP_ON_ERROR):
        expected_encoding = os.path.abspath('expected_encoding.json.gz')
        expected_rows = os.path.abspath('expected_rows.pbs.gz')
        actual_encoding = os.path.abspath('actual_encoding.json.gz')
        actual_rows = os.path.abspath('actual_rows.pbs.gz')
        with mock.patch('loom.util.THREADS', new_value=1):
            loom.format.make_encoding(
                schema_in=schema,
                rows_in=rows_csv,
                encoding_out=expected_encoding)
            loom.format.import_rows(
                encoding_in=expected_encoding,
                rows_in=rows_csv,
                rows_out=expected_rows)
        with mock.patch('loom.util.THREADS', new_value=8):
            loom.format.make_encoding(
                schema_in=schema,
                rows_in=rows_csv,
                encoding_out=actual_encoding)
            loom.format.import_rows(
                encoding_in=actual_encoding,
                rows_in=rows_csv,
                rows_out=actual_rows)
        assert_equal(json_load(actual_encoding), json_load(expected_encoding))
        for actual, expected in izip(
                protobuf_stream_load(actual_rows),
                protobuf_stream_load(expected_rows)):
            assert_close(actual, expected)
