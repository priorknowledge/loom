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
from distributions.fileutil import tempdir
import loom.format
import loom.generate
import loom.config
import loom.runner
import loom.query
from loom.test.util import for_each_dataset, CLEANUP_ON_ERROR, assert_found
from loom.test.test_query import get_example_requests


def make_config(config_out, seed=0):
    config = {
        'schedule': {'extra_passes': 2},
        'seed': seed,
    }
    loom.config.config_dump(config, config_out)


@for_each_dataset
def test_all(schema, rows_csv, **unused):
    seed = 7
    with tempdir(cleanup_on_error=CLEANUP_ON_ERROR):
        encoding = os.path.abspath('encoding.json.gz')
        rows = os.path.abspath('rows.pbs.gz')
        shuffled = os.path.abspath('shuffled.pbs.gz')
        init = os.path.abspath('init.pb.gz')
        config = os.path.abspath('config.pb.gz')
        model = os.path.abspath('model.pb.gz')
        groups = os.path.abspath('groups')
        assign = os.path.abspath('assign.pbs.gz')
        log = os.path.abspath('log.pbs.gz')
        os.mkdir(groups)

        print 'making encoding'
        loom.format.make_encoding(
            schema_in=schema,
            rows_in=rows_csv,
            encoding_out=encoding)
        assert_found(encoding)

        print 'importing rows'
        loom.format.import_rows(
            encoding_in=encoding,
            rows_csv_in=rows_csv,
            rows_out=rows)
        assert_found(rows)

        print 'generating init'
        loom.generate.generate_init(
            encoding_in=encoding,
            model_out=init,
            seed=seed)
        assert_found(init)

        print 'shuffling rows'
        loom.runner.shuffle(
            rows_in=rows,
            rows_out=shuffled,
            seed=seed)

        print 'creating config'
        make_config(
            config_out=config,
            seed=seed)
        assert_found(config)

        print 'inferring'
        loom.runner.infer(
            config_in=config,
            rows_in=shuffled,
            model_in=init,
            model_out=model,
            groups_out=groups,
            assign_out=assign,
            log_out=log,
            debug=True)
        assert_found(model, groups, assign, log)

        print 'querying'
        requests = get_example_requests(model)
        server = loom.query.SingleSampleProtobufServer(
            config_in=config,
            model_in=model,
            groups_in=groups,
            debug=True)
        for req in requests:
            server.send(req)
            server.receive()
