import os
import loom.runner
from distributions.fileutil import tempdir
from distributions.io.stream import protobuf_stream_load, protobuf_stream_dump
from loom.schema_pb2 import Query

serve = loom.runner.query.serve


def parse_result(message):
    result = Query.Response()
    result.ParseFromString(message)
    return result


def batch_predict(
        config_in,
        model_in,
        groups_in,
        queries,
        debug=False,
        profile=None):
    root = os.path.abspath(os.path.curdir)
    with tempdir(cleanup_on_error=(not debug)):
        queries_in = os.path.abspath('queries.pbs.gz')
        results_out = os.path.abspath('results.pbs.gz')
        protobuf_stream_dump(
            (q.SerializeToString() for q in queries),
            queries_in)

        os.chdir(root)
        loom.runner.query(
            config_in=config_in,
            model_in=model_in,
            groups_in=groups_in,
            queries_in=queries_in,
            results_out=results_out,
            debug=debug,
            profile=profile)

        return map(parse_result, protobuf_stream_load(results_out))
