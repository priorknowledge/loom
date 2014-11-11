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
import numpy.random
from itertools import izip
from distributions.fileutil import tempdir
import loom.store
import loom.transforms
import loom.tasks
from loom.transforms import EXAMPLE_VALUES


def generate_cell(possible_values, observed_prob=0.5):
    if numpy.random.random() <= observed_prob:
        return numpy.random.choice(possible_values)
    else:
        return None


def generate_example(schema_csv, rows_csv, row_count=100):
    feature_names = []
    fluent_types = []
    for fluent_type in EXAMPLE_VALUES.iterkeys():
        if fluent_type != 'id':
            fluent_types.append(fluent_type)
            feature_names.append('{}_feature'.format(fluent_type))

    with loom.util.csv_writer(schema_csv) as writer:
        writer.writerow(['Feature Name', 'Type'])
        writer.writerow(['id', 'id'])
        writer.writerows(izip(feature_names, fluent_types))

    with loom.util.csv_writer(rows_csv) as writer:
        writer.writerow(['id'] + feature_names)
        for row_id in xrange(row_count):
            values = [
                generate_cell(EXAMPLE_VALUES[fluent_type])
                for fluent_type in fluent_types
            ]
            writer.writerow([row_id] + values)


def test_transforms():
    name = 'test_transforms.test_transforms'
    with tempdir() as temp:
        schema_csv = os.path.join(temp, 'schema.csv')
        rows_csv = os.path.join(temp, 'rows.csv.gz')
        generate_example(schema_csv, rows_csv)
        loom.tasks.transform(name, schema_csv, rows_csv)
        loom.tasks.ingest(name)
        loom.tasks.infer(name, sample_count=1)
