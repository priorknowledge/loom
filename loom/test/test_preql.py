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
import csv
from nose.tools import (
    assert_almost_equal,
    assert_equal,
    assert_true,
)
import numpy
from distributions.fileutil import tempdir
from distributions.io.stream import open_compressed, json_load
from distributions.tests.util import assert_close
import loom.preql
import loom.query
from loom.format import load_encoder
from loom.test.util import for_each_dataset, CLEANUP_ON_ERROR

COUNT = 10


def _check_predictions(rows_in, result_out, name_to_encoder):
    with open_compressed(rows_in, 'rb') as fin:
        with open(result_out, 'r') as fout:
            in_reader = csv.reader(fin)
            out_reader = csv.reader(fout)
            fnames = in_reader.next()
            out_reader.next()
            for in_row in in_reader:
                for i in range(COUNT):
                    out_row = out_reader.next()
                    bundle = zip(fnames, in_row, out_row)
                    for name, in_val, out_val in bundle:
                        if name == '_id':
                            assert_equal(in_val, out_val)
                            continue
                        encode = name_to_encoder[name]
                        observed = bool(in_val.strip())
                        if observed:
                            assert_almost_equal(
                                encode(in_val),
                                encode(out_val))
                        else:
                            assert_true(bool(out_val.strip()))


@for_each_dataset
def test_predict(root, rows_csv, encoding, **unused):
    with tempdir(cleanup_on_error=CLEANUP_ON_ERROR):
        with loom.query.get_server(root, debug=True) as query_server:
            result_out = 'predictions_out.csv'
            rows_in = os.listdir(rows_csv)[0]
            rows_in = os.path.join(rows_csv, rows_in)
            encoders = json_load(encoding)
            name_to_encoder = {e['name']: load_encoder(e) for e in encoders}
            preql = loom.preql.PreQL(query_server, encoding)
            preql.predict(rows_in, COUNT, result_out, id_offset=False)
            _check_predictions(rows_in, result_out, name_to_encoder)


@for_each_dataset
def test_predict_ids(root, rows_csv, encoding, **unused):
    with tempdir(cleanup_on_error=CLEANUP_ON_ERROR):
        with loom.query.get_server(root, debug=True) as query_server:
            result_out = 'predictions_out.csv'
            rows_in = os.listdir(rows_csv)[0]
            rows_in = os.path.join(rows_csv, rows_in)
            id_rows_in = 'rows_in.csv'
            with open_compressed(rows_in) as fin:
                with open(id_rows_in, 'w') as fout:
                    reader = csv.reader(fin)
                    writer = csv.writer(fout)
                    header = reader.next()
                    header.insert(0, '_id')
                    writer.writerow(header)
                    for i, row in enumerate(reader):
                        row.insert(0, 'o{}'.format(i))
                        writer.writerow(row)
            encoders = json_load(encoding)
            name_to_encoder = {e['name']: load_encoder(e) for e in encoders}
            preql = loom.preql.PreQL(query_server, encoding)
            preql.predict(id_rows_in, COUNT, result_out, id_offset=True)
            _check_predictions(id_rows_in, result_out, name_to_encoder)


@for_each_dataset
def test_relate(root, encoding, **unused):
    with loom.query.get_server(root, debug=True) as query_server:
        with tempdir(cleanup_on_error=CLEANUP_ON_ERROR):
            result_out = 'related_out.csv'
            preql = loom.preql.PreQL(query_server, encoding)
            preql.relate(preql.feature_names, result_out, sample_count=10)
            with open(result_out, 'r') as f:
                reader = csv.reader(f)
                header = reader.next()
                columns = header[1:]
                assert_equal(columns, preql.feature_names)
                zmatrix = numpy.zeros((len(columns), len(columns)))
                for i, row in enumerate(reader):
                    column = row.pop(0)
                    assert_equal(column, preql.feature_names[i])
                    for j, score in enumerate(row):
                        score = float(score)
                        zmatrix[i][j] = score
                assert_close(zmatrix, zmatrix.T)


@for_each_dataset
def test_group_runs(root, schema, encoding, **unused):
    with tempdir(cleanup_on_error=CLEANUP_ON_ERROR):
        with loom.preql.get_server(root, encoding, debug=True) as preql:
            test_columns = json_load(schema).keys()[:10]
            for column in test_columns:
                groupings_csv = 'group.{}.csv'.format(column)
                preql.group(column, result_out=groupings_csv)
                print open(groupings_csv).read()
