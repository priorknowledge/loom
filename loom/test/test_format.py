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
