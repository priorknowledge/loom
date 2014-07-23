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

from distributions.io.stream import protobuf_stream_load
import loom.util
from loom.util import mkdir_p, rm_rf
import loom.format
import loom.generate
import loom.config
import loom.runner
import loom.query
from loom.test.util import for_each_dataset, assert_found
from loom.test.test_query import get_example_requests


def make_config(config_out, seed=0):
    config = {
        'schedule': {'extra_passes': 2},
        'seed': seed,
    }
    loom.config.config_dump(config, config_out)


@for_each_dataset
def test_all(name, schema, rows_csv, **unused):
    results = loom.store.get_paths(name, 'test_happy_path')
    rm_rf(results['root'])
    mkdir_p(results['root'])

    seed = 7

    print 'creating config'
    make_config(
        config_out=results['config'],
        seed=seed)
    assert_found(results['config'])

    print 'making schema row'
    loom.format.make_schema_row(
        schema_in=schema,
        schema_row_out=results['schema_row'])
    assert_found(results['schema_row'])

    print 'making encoding'
    loom.format.make_encoding(
        schema_in=schema,
        rows_in=rows_csv,
        encoding_out=results['encoding'])
    assert_found(results['encoding'])

    print 'importing rows'
    loom.format.import_rows(
        encoding_in=results['encoding'],
        rows_csv_in=rows_csv,
        rows_out=results['rows'])
    assert_found(results['rows'])

    print 'making tare rows'
    loom.runner.tare(
        schema_row_in=results['schema_row'],
        rows_in=results['rows'],
        tares_out=results['tares'],
        debug=True)
    assert_found(results['tares'])

    tare_count = sum(1 for _ in protobuf_stream_load(results['tares']))
    print 'sparsifying rows WRT {} tare rows'.format(tare_count)
    loom.runner.sparsify(
        schema_row_in=results['schema_row'],
        tares_in=results['tares'],
        rows_in=results['rows'],
        rows_out=results['diffs'],
        debug=True)
    assert_found(results['diffs'])

    print 'generating init'
    loom.generate.generate_init(
        encoding_in=results['encoding'],
        model_out=results['init'],
        seed=seed)
    assert_found(results['init'])

    print 'shuffling rows'
    loom.runner.shuffle(
        rows_in=results['diffs'],
        rows_out=results['shuffled'],
        seed=seed,
        debug=True)
    assert_found(results['shuffled'])

    print 'inferring'
    loom.runner.infer(
        config_in=results['config'],
        rows_in=results['shuffled'],
        tares_in=results['tares'],
        model_in=results['init'],
        model_out=results['model'],
        groups_out=results['groups'],
        assign_out=results['assign'],
        log_out=results['infer_log'],
        debug=True)
    assert_found(results['model'], results['groups'], results['assign'])
    assert_found(results['infer_log'])

    print 'querying'
    requests = get_example_requests(results['model'], results['rows'])
    server = loom.query.SingleSampleProtobufServer(
        config_in=results['config'],
        model_in=results['model'],
        groups_in=results['groups'],
        debug=True)
    for req in requests:
        server.send(req)
        server.receive()
