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
from loom.util import LOG
import loom.store
import loom.datasets
import loom.tasks
import loom.query
from loom.test.util import for_each_dataset
from loom.test.test_query import get_example_requests, check_response

SAMPLE_COUNT = 2
CONFIG = {'schedule': {'extra_passes': 2}}


@for_each_dataset
def test_all(name, schema, rows_csv, **unused):
    name = os.path.join(name, 'test_tasks')
    paths = loom.store.get_paths(name)
    loom.datasets.clean(name)
    loom.tasks.ingest(name, schema, rows_csv, debug=True)
    loom.tasks.infer(
        name,
        sample_count=SAMPLE_COUNT,
        config=CONFIG,
        debug=True)
    loom.tasks.get_consensus(name, debug=True)

    LOG('querying')
    requests = get_example_requests(
        paths['samples'][0]['model'],
        paths['ingest']['rows'])
    Server = loom.query.MultiSampleProtobufServer
    with Server(paths['samples'], debug=True) as server:
        for request in requests:
            server.send(request)
            response = server.receive()
            check_response(request, response)
