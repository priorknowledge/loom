import os
import sys
import subprocess
import loom.runner
from distributions.fileutil import tempdir
from distributions.io.stream import (
    open_compressed,
    protobuf_stream_write,
    protobuf_stream_read,
    protobuf_stream_load,
    protobuf_stream_dump,
)
from loom.schema_pb2 import CrossCat, PreQL

PYTHON = sys.executable


def parse_result(message):
    result = PreQL.Predict.Result()
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
        loom.runner.predict(
            config_in=config_in,
            model_in=model_in,
            groups_in=groups_in,
            queries_in=queries_in,
            results_out=results_out,
            debug=debug,
            profile=profile)

        return map(parse_result, protobuf_stream_load(results_out))


class Server(object):
    def __init__(self, config_in, model_in, groups_in, debug=False):
        args = [
            PYTHON, '-m', 'loom.runner', 'predict',
            'config_in={}'.format(config_in),
            'model_in={}'.format(model_in),
            'groups_in={}'.format(groups_in),
            'queries_in=-',
            'results_out=-',
            'debug={}'.format(debug),
        ]
        self.devnull = open(os.devnull, 'w')
        self.proc = subprocess.Popen(
            args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.devnull)

    def __call__(self, query):
        string = query.SerializeToString()
        protobuf_stream_write(string, self.proc.stdin)
        string = protobuf_stream_read(self.proc.stdout)
        result = PreQL.Predict.Result()
        result.ParseFromString(string)
        return result

    def close(self):
        self.proc.stdin.close()
        self.proc.wait()
        self.devnull.close()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()


def get_example_queries(model):
    cross_cat = CrossCat()
    with open_compressed(model) as f:
        cross_cat.ParseFromString(f.read())
    feature_count = sum(len(kind.featureids) for kind in cross_cat.kinds)

    all_observed = [True] * feature_count
    none_observed = [False] * feature_count
    observeds = []
    observeds.append(all_observed)
    for f in xrange(feature_count):
        observed = all_observed[:]
        observed[f] = False
        observeds.append(observed)
    for f in xrange(feature_count):
        observed = none_observed[:]
        observed[f] = True
        observeds.append(observed)
    observeds.append(none_observed)

    queries = []
    for i, observed in enumerate(observeds):
        query = PreQL.Predict.Query()
        query.id = "example-{}".format(i)
        query.data.observed[:] = none_observed
        query.to_predict[:] = observed
        query.sample_count = 1
        queries.append(query)

    return queries
