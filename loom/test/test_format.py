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
from nose.tools import assert_true
from distributions.fileutil import tempdir
import loom.format
from loom.test.util import for_each_dataset, CLEANUP_ON_ERROR


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
        assert_true(os.path.exists(schema_out))
        assert_true(os.path.exists(encoding_out))


@for_each_dataset
def test_export_import_rows(model, rows, **unused):
    with tempdir(cleanup_on_error=CLEANUP_ON_ERROR):
        schema_json = os.path.abspath('schema.json.gz')
        encoding_json = os.path.abspath('encoding.json.gz')
        rows_csv = os.path.abspath('rows.csv.gz')
        rows_pbs = os.path.abspath('rows.pbs.gz')
        loom.format.make_fake_encoding(
            model_in=model,
            rows_in=rows,
            schema_out=schema_json,
            encoding_out=encoding_json)
        assert_true(os.path.exists(schema_json))
        assert_true(os.path.exists(encoding_json))
        loom.format.export_rows(encoding_json, rows, rows_csv)
        assert_true(os.path.exists(rows_csv))
        os.remove(encoding_json)
        loom.format.make_encoding(schema_json, rows_csv, encoding_json)
        assert_true(os.path.exists(encoding_json))
        loom.format.import_rows(encoding_json, rows_csv, rows_pbs)
        assert_true(os.path.exists(rows_pbs))
