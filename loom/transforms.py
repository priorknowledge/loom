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

from itertools import izip
from contextlib2 import ExitStack
from distributions.io.stream import json_load
import parsable
import loom.util
parsable = parsable.Parsable()


def TODO(message):
    raise NotImplementedError('TODO {}'.format(message))


class Transform(object):
    def dump(self):
        return None


def load_transform(filename):
    return TODO(filename)


@parsable.command
def make_transforms(schema_in):
    basic_schema = json_load(schema_in)
    TODO(basic_schema)


@parsable.command
def transform_schema(transforms_in, schema_in, schema_out):
    raise NotImplementedError()


@parsable.command
def transform_rows(transforms_in, rows_in, rows_out):
    transforms = {
        key: load_transform(val)
        for key, val in json_load(transforms_in)
    }
    with ExitStack() as stack:
        with_ = stack.enter_context
        reader = with_(loom.util.csv_reader(rows_in))
        writer = with_(loom.util.csv_writer(rows_out))
        header = reader.next()
        basic_header = TODO(header)
        writer.writerow(basic_header)
        for row in reader:
            row_dict = {
                key: val
                for key, val in izip(header, row)
                if key is not None
            }
            writer.writerow(transforms[key](row_dict) for key in basic_header)


if __name__ == '__main__':
    parsable.dispatch()
