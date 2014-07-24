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
from loom.test.test_query import get_example_requests, check_response

SAMPLE_COUNT = 2


def make_config(config_out, seed=0):
    config = {
        'schedule': {'extra_passes': 2},
        'seed': seed,
    }
    loom.config.config_dump(config, config_out)


@for_each_dataset
def test_all(name, schema, rows_csv, **unused):
    results = loom.store.get_paths(name, 'test_happy_path')
    samples = loom.store.get_samples(name, 'test_happy_path', SAMPLE_COUNT)
    rm_rf(results['root'])
    mkdir_p(results['root'])

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

    for seed, sample in enumerate(samples):

        print 'generating init'
        loom.generate.generate_init(
            encoding_in=sample['encoding'],
            model_out=sample['init'],
            seed=seed)
        assert_found(sample['init'])

        print 'shuffling rows'
        loom.runner.shuffle(
            rows_in=sample['diffs'],
            rows_out=sample['shuffled'],
            seed=seed,
            debug=True)
        assert_found(sample['shuffled'])

        print 'creating config'
        make_config(
            config_out=sample['config'],
            seed=seed)
        assert_found(sample['config'])

        print 'inferring'
        loom.runner.infer(
            config_in=sample['config'],
            rows_in=sample['shuffled'],
            tares_in=sample['tares'],
            model_in=sample['init'],
            model_out=sample['model'],
            groups_out=sample['groups'],
            assign_out=sample['assign'],
            log_out=sample['infer_log'],
            debug=True)
        assert_found(sample['model'], sample['groups'], sample['assign'])
        assert_found(sample['infer_log'])

    print 'querying'
    requests = get_example_requests(results['model'], results['rows'])
    with loom.query.get_protobuf_server(results['root']) as server:
        for request in requests:
            server.send(request)
            response = server.receive()
            check_response(request, response)
