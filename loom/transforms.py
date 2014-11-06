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
import re
import dateutil.parser
from itertools import izip
from collections import Counter
from contextlib2 import ExitStack
from distributions.io.stream import json_dump
from distributions.io.stream import json_load
import loom.util
from loom.util import cp_ns
from loom.util import parallel_map
from loom.util import pickle_dump
from loom.util import pickle_load
import loom.documented
import parsable
parsable = parsable.Parsable()

EXAMPLE_VALUES = {
    'boolean': ['0', '1', 'true', 'false'],
    'categorical': ['Monday', 'June'],
    'unbounded_categorical': ['CRM', '90210'],
    'count': ['0', '1', '2', '3', '4'],
    'real': ['-100.0', '1e-4'],
    'sparse_real': ['0', '0', '0', '0', '123456.78', '0', '0', '0'],
    'date': ['2014-03-31', '10pm, August 1, 1979'],
    'text': ['This is a text feature.', 'Hello World!'],
    'tags': ['', 'big_data machine_learning platform'],
}
for fluent_type, values in EXAMPLE_VALUES.items():
    EXAMPLE_VALUES['optional_{}'.format(fluent_type)] = [''] + values
EXAMPLE_VALUES['id'] = ['any unique string can serve as an id']

FLUENT_TO_BASIC = {
    'boolean': 'bb',
    'categorical': 'dd',
    'unbounded_categorical': 'dpd',
    'count': 'gp',
    'real': 'nich',
}


def get_row_dict(header, row):
    '''By convention, empty strings are omitted from the result dict.'''
    return {key: value for key, value in izip(header, row) if value}


# ----------------------------------------------------------------------------
# simple transforms

class StringTransform(object):
    def __init__(self, feature_name, fluent_type):
        self.feature_name = feature_name
        self.basic_type = FLUENT_TO_BASIC[fluent_type]

    def get_schema(self):
        return {self.feature_name: self.basic_type}

    def __call__(self, row_dict):
        feature_name = self.feature_name
        if feature_name in row_dict:
            row_dict[feature_name] = row_dict[feature_name].lower()


class PercentTransform(object):
    def __init__(self, feature_name):
        self.feature_name = feature_name

    def get_schema(self):
        return {self.feature_name: 'nich'}

    def __call__(self, row_dict):
        feature_name = self.feature_name
        if feature_name in row_dict:
            value = float(row_dict[feature_name].replace('%', '')) * 0.01
            row_dict[feature_name] = value


class PresenceTransform(object):
    def __init__(self, feature_name):
        self.feature_name = feature_name
        self.present_name = '{}.present'.format(feature_name)
        self.value_name = '{}.value'.format(feature_name)

    def get_schema(self):
        return {self.present_name: 'bb'}

    def __call__(self, row_dict):
        present = (self.feature_name in row_dict)
        row_dict[self.present_name] = present
        if present:
            row_dict[self.value_name] = row_dict[self.feature_name]


class SparseRealTransform(object):
    def __init__(self, feature_name, tare_value=0.0):
        self.feature_name = feature_name
        self.nonzero_name = '{}.nonzero'.format(feature_name)
        self.value_name = '{}.value'.format(feature_name)
        self.tare_value = float(tare_value)

    def get_schema(self):
        return {self.nonzero_name: 'bb', self.value_name: 'nich'}

    def __call__(self, row_dict):
        feature_name = self.feature_name
        if feature_name in row_dict:
            value = float(row_dict[feature_name])
            nonzero = (value == self.tare_value)
            row_dict[self.nonzero_name] = nonzero
            if nonzero:
                row_dict[self.value_name] = value


# ----------------------------------------------------------------------------
# text transform

MIN_WORD_FREQ = 0.01

split_text = re.compile('\W+').split


def get_word_set(text):
    return frozenset(s for s in split_text(text.lower()) if s)


class TextTransformBuilder(object):
    def __init__(
            self,
            feature_name,
            allow_empty=False,
            min_word_freq=MIN_WORD_FREQ):
        self.feature_name = feature_name
        self.counts = Counter()
        self.min_word_freq = min_word_freq
        self.allow_empty = allow_empty

    def add_row(self, row_dict):
        text = row_dict.get(self.feature_name, '')
        self.counts.update(get_word_set(text))

    def build(self):
        counts = self.counts.most_common()
        max_count = counts[0][1]
        min_count = self.min_word_freq * max_count
        words = [word for word, count in counts if count > min_count]
        return TextTransform(self.feature_name, words, self.allow_empty)


class TextTransform(object):
    def __init__(self, feature_name, words, allow_empty):
        self.feature_name = feature_name
        self.features = [
            ('{}.{}'.format(feature_name, word), word)
            for word in words
        ]
        self.allow_empty = allow_empty

    def get_schema(self):
        return {feature_name: 'bb' for feature_name, _ in self.features}

    def __call__(self, row_dict):
        if self.feature_name in row_dict or self.allow_empty:
            text = row_dict.get(self.feature_name, '')
            word_set = get_word_set(text)
            row_dict.pop(self.feature_name, None)
            for feature_name, word in self.features:
                row_dict[feature_name] = (word in word_set)


# ----------------------------------------------------------------------------
# date transform

EPOCH = dateutil.parser.parse('2014-03-31')  # arbitrary (Loom's birthday)


def days_between(start, end):
    return (end - start).total_seconds() / (24 * 60 * 60)


class DateTransform(object):
    def __init__(self, feature_name, relatives):
        self.feature_name = feature_name
        self.relatives = relatives
        suffices = ['absolute', 'mod.year', 'mod.month', 'mod.week', 'mod.day']
        self.abs_names = {
            suffix: '{}.{}'.format(feature_name, suffix)
            for suffix in suffices
        }
        self.rel_names = {
            relative: '{}.minus.{}'.format(feature_name, relative)
            for relative in relatives
        }

    def get_schema(self):
        schema = {
            self.abs_names['absolute']: 'nich',
            self.abs_names['mod.year']: 'dpd',
            self.abs_names['mod.month']: 'dpd',
            self.abs_names['mod.week']: 'dpd',
            self.abs_names['mod.day']: 'dpd',
        }
        for rel_name in self.rel_names.itervalues():
            schema[rel_name] = 'nich'
        return schema

    def __call__(self, row_dict):
        if self.feature_name in row_dict:
            date = dateutil.parser.parse(row_dict[self.feature_name])

            abs_names = self.abs_names
            row_dict[abs_names['absolute']] = days_between(EPOCH, date)
            row_dict[abs_names['mod.year']] = date.month
            row_dict[abs_names['mod.month']] = date.day
            row_dict[abs_names['mod.week']] = date.weekday()
            row_dict[abs_names['mod.day']] = date.hour

            for relative, rel_name in self.rel_names.iteritems():
                if relative in row_dict:
                    other_date = dateutil.parser.parse(row_dict[relative])
                    row_dict[rel_name] = days_between(other_date, date)


# ----------------------------------------------------------------------------
# commands

def build_transforms(rows_in, transforms, builders):
    if os.path.isdir(rows_in):
        filenames = [os.path.join(rows_in, f) for f in os.listdir(rows_in)]
    else:
        filenames = [rows_in]
    for filename in filenames:
        with loom.util.csv_reader(filename) as reader:
            header = reader.next()
            for row in reader:
                row_dict = get_row_dict(header, row)
                for transform in transforms:
                    transform(row_dict)
                for builder in builders:
                    builder.add_row(row_dict)
    return [builder.build() for builder in builders]


@loom.documented.transform(
    inputs=['schema', 'rows_csv'],
    outputs=['ingest.schema', 'ingest.transforms'])
@parsable.command
def make_transforms(schema_in, rows_in, schema_out, transforms_out):
    fluent_schema = json_load(schema_in)
    fluent_schema = {k: v for k, v in fluent_schema.iteritems() if v}
    basic_schema = {}
    pre_transforms = []
    transforms = []
    builders = []
    dates = [
        feature_name
        for feature_name, fluent_type in fluent_schema.iteritems()
        if fluent_type.endswith('date')
    ]
    id_field = None
    for feature_name, fluent_type in fluent_schema.iteritems():
        # parse adjectives
        if fluent_type.startswith('optional_'):
            transform = PresenceTransform(feature_name)
            pre_transforms.append(transform)
            transforms.append(transform)
            fluent_type = fluent_type[len('optional_'):]
            feature_name = '{}.value'.format(feature_name)

        # parse nouns
        if fluent_type == 'id':
            id_field = feature_name
        elif fluent_type in ['categorical', 'unbounded_categorical']:
            transforms.append(StringTransform(feature_name, fluent_type))
        elif fluent_type == 'percent':
            transforms.append(PercentTransform(feature_name))
        elif fluent_type == 'sparse_real':
            transforms.append(SparseRealTransform(feature_name))
        elif fluent_type == 'text':
            builders.append(TextTransformBuilder(feature_name))
        elif fluent_type == 'tags':
            builders.append(
                TextTransformBuilder(feature_name, allow_empty=True))
        elif fluent_type == 'date':
            relatives = [other for other in dates if other < feature_name]
            transforms.append(DateTransform(feature_name, relatives))
        else:
            basic_type = FLUENT_TO_BASIC[fluent_type]
            basic_schema[feature_name] = basic_type
    if builders:
        transforms += build_transforms(rows_in, pre_transforms, builders)
    for transform in transforms:
        basic_schema.update(transform.get_schema())
    json_dump(basic_schema, schema_out)
    pickle_dump(transforms, transforms_out)
    return id_field


def _transform_rows((transforms, transformed_header, rows_in, rows_out)):
    with ExitStack() as stack:
        with_ = stack.enter_context
        reader = with_(loom.util.csv_reader(rows_in))
        writer = with_(loom.util.csv_writer(rows_out))
        header = reader.next()
        writer.writerow(transformed_header)
        for row in reader:
            row_dict = get_row_dict(header, row)
            for transform in transforms:
                transform(row_dict)
            writer.writerow([row_dict.get(key) for key in transformed_header])


@loom.documented.transform(
    inputs=['ingest.schema', 'ingest.transforms', 'rows_csv'],
    outputs=['ingest.rows_csv'])
@parsable.command
def transform_rows(schema_in, transforms_in, rows_in, rows_out, id_field=None):
    transforms = pickle_load(transforms_in)
    if not transforms:
        cp_ns(rows_in, rows_out)
    else:
        transformed_header = sorted(json_load(schema_in).iterkeys())
        if id_field is not None:
            assert id_field not in transformed_header
            transformed_header = [id_field] + transformed_header
        tasks = []
        if os.path.isdir(rows_in):
            loom.util.mkdir_p(rows_out)
            for f in os.listdir(rows_in):
                tasks.append((
                    transforms,
                    transformed_header,
                    os.path.join(rows_in, f),
                    os.path.join(rows_out, f),
                ))
        else:
            tasks.append((transforms, transformed_header, rows_in, rows_out))
        parallel_map(_transform_rows, tasks)


def make_fake_transforms(transforms_out):
    pickle_dump([], transforms_out)


if __name__ == '__main__':
    parsable.dispatch()
