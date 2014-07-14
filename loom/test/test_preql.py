import loom.preql
import loom.query
from loom.format import load_encoder
from test_query import get_protobuf_server
from distributions.fileutil import tempdir
from distributions.io.stream import open_compressed, json_load
from loom.test.util import for_each_dataset, CLEANUP_ON_ERROR
import os
import csv
from nose.tools import assert_true, assert_almost_equal


@for_each_dataset
def test_predict(model, groups, rows_csv, encoding, **unused):
    COUNT = 10
    with tempdir(cleanup_on_error=CLEANUP_ON_ERROR):
        protobuf_server = get_protobuf_server(
            loom.query.MultiSampleProtobufServer,
            [model, model],
            [groups, groups])
        result_out = 'predictions_out.csv'
        rows_in = os.listdir(rows_csv)[0]
        rows_in = os.path.join(rows_csv, rows_in)
        encoders = json_load(encoding)
        name_to_encoder = {e['name']: load_encoder(e) for e in encoders}
        query_server = loom.query.QueryServer(protobuf_server)
        preql = loom.preql.PreQL(query_server, encoding)
        preql.predict(rows_in, COUNT, result_out, id_offset=False)
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
                            encode = name_to_encoder[name]
                            observed = bool(in_val.strip())
                            if observed:
                                assert_almost_equal(
                                    encode(in_val),
                                    encode(out_val))
                            else:
                                assert_true(bool(out_val.strip()))
