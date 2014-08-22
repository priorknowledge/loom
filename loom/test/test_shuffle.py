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
from nose.tools import assert_equal, assert_not_equal, assert_list_equal
from loom.test.util import (
    for_each_dataset,
    CLEANUP_ON_ERROR,
    assert_found,
    load_rows,
    load_rows_raw,
)
from distributions.fileutil import tempdir
import loom.runner


@for_each_dataset
def test_one_to_one(rows, **unused):
    with tempdir(cleanup_on_error=CLEANUP_ON_ERROR):
        seed = 12345
        rows_out = os.path.abspath('rows_out.pbs.gz')
        loom.runner.shuffle(
            rows_in=rows,
            rows_out=rows_out,
            seed=seed)
        assert_found(rows_out)

        original = load_rows(rows)
        shuffled = load_rows(rows_out)
        assert_equal(len(shuffled), len(original))
        assert_not_equal(shuffled, original)

        actual = sorted(shuffled, key=lambda row: row.id)
        expected = sorted(original, key=lambda row: row.id)
        assert_list_equal(expected, actual)


@for_each_dataset
def test_chunking(rows, **unused):
    targets = [10.0 ** i for i in xrange(6)]
    with tempdir(cleanup_on_error=CLEANUP_ON_ERROR):
        seed = 12345
        rows_out = os.path.abspath('rows.out.{}.pbs.gz')

        for i, target in enumerate(targets):
            loom.runner.shuffle(
                rows_in=rows,
                rows_out=rows_out.format(i),
                seed=seed,
                target_mem_bytes=target)

        results = [
            load_rows_raw(rows_out.format(i))
            for i in xrange(len(targets))
        ]
        for i, actual in enumerate(results):
            for expected in results[:i]:
                assert_list_equal(actual, expected)
