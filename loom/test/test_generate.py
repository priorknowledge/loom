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
from loom.test.util import for_each_dataset, CLEANUP_ON_ERROR, assert_found
import loom.generate

FEATURE_TYPES = loom.schema.FEATURE_TYPES.keys()
FEATURE_TYPES += ['mixed']


def test_generate():
    for feature_type in FEATURE_TYPES:
        yield _test_generate, feature_type


def _test_generate(feature_type):
    root = os.getcwd()
    with tempdir(cleanup_on_error=CLEANUP_ON_ERROR):
        init_out = os.path.abspath('init.pb.gz')
        rows_out = os.path.abspath('rows.pbs.gz')
        model_out = os.path.abspath('model.pb.gz')
        groups_out = os.path.abspath('groups')
        os.chdir(root)
        loom.generate.generate(
            feature_type=feature_type,
            row_count=100,
            feature_count=100,
            density=0.5,
            init_out=init_out,
            rows_out=rows_out,
            model_out=model_out,
            groups_out=groups_out,
            debug=True,
            profile=None)


@for_each_dataset
def test_generate_init(encoding, **unused):
    with tempdir(cleanup_on_error=CLEANUP_ON_ERROR):
        init_out = os.path.abspath('init.pb.gz')
        loom.generate.generate_init(
            encoding_in=encoding,
            model_out=init_out)
        assert_found(init_out)
