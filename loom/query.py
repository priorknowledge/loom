import os
import loom.runner
from distributions.fileutil import tempdir
from distributions.io.stream import protobuf_stream_load, protobuf_stream_dump
from loom.schema_pb2 import Query

serve = loom.runner.query.serve


def parse_response(message):
    response = Query.Response()
    response.ParseFromString(message)
    return response


def batch_predict(
        config_in,
        model_in,
        groups_in,
        requests,
        debug=False,
        profile=None):
    root = os.path.abspath(os.path.curdir)
    with tempdir(cleanup_on_error=(not debug)):
        requests_in = os.path.abspath('requests.pbs.gz')
        responses_out = os.path.abspath('responses.pbs.gz')
        protobuf_stream_dump(
            (q.SerializeToString() for q in requests),
            requests_in)

        os.chdir(root)
        loom.runner.query(
            config_in=config_in,
            model_in=model_in,
            groups_in=groups_in,
            requests_in=requests_in,
            responses_out=responses_out,
            debug=debug,
            profile=profile)

        return map(parse_response, protobuf_stream_load(responses_out))
