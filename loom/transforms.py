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

import re
import dateutil.parser
from itertools import izip
from collections import Counter
from contextlib2 import ExitStack
from distributions.io.stream import json_dump
from distributions.io.stream import json_load
import loom.util
from loom.util import cp_ns
from loom.util import pickle_dump
from loom.util import pickle_load
import loom.documented
import parsable
parsable = parsable.Parsable()


def TODO(message):
    raise NotImplementedError('TODO {}'.format(message))


FEATURE_FREQ = 0.01

FLUENT_TYPES = {
    'boolean': 'bb',
    'count': 'gp',
    'categorical': 'dd',
    'unbounded_categorical': 'dpd',
    'real': 'nich',
    'sparse_real': 'nich',
    'text': None,
    'date': None,
}


def get_row_dict(header, row):
    '''By convention, empty strings are omitted from the result dict.'''
    return {key: value for key, value in izip(header, row) if value}


# ----------------------------------------------------------------------------
# simple transforms

def PresenceTransform(object):
    def __init__(self, feature_name):
        self.feature_name = feature_name
        self.present_name = '{}.present'.format(self.feature_name)

    def get_schema(self):
        return {self.present_name: 'bb'}

    def __call__(self, row, transformed_row):
        transformed_row[self.present_name] = (self.feature_name in row)


def SparseRealTransform(object):
    def __init__(self, feature_name, tare_value=0.0):
        self.feature_name = feature_name
        self.nonzero_name = '{}.nonzero'.format(feature_name)
        self.value_name = '{}.value'.format(feature_name)
        self.tare_value = float(tare_value)

    def get_schema(self):
        return {self.nonzero_name: 'bb', self.value_name: 'nich'}

    def __call__(self, row, transformed_row):
        feature_name = self.feature_name
        if feature_name in row:
            del transformed_row[feature_name]
            value = float(row['feature_name'])
            nonzero = (value == self.tare_value)
            transformed_row[self.nonzero_name] = nonzero
            if nonzero:
                transformed_row[self.value_name] = value


# ----------------------------------------------------------------------------
# text transforms

# TODO internationalize
re_nonalpha = re.compile('[^a-z]+')


def get_word_set(text):
    return frozenset(s for s in re_nonalpha.split(text.lower()) if s)


class TextTransformBuilder(object):
    def __init__(self, feature_name, feature_freq=FEATURE_FREQ):
        self.feature_name = feature_name
        self.counts = Counter()
        self.feature_freq = feature_freq

    def add_row(self, row_dict):
        text = row_dict[self.feature_name]
        self.counter.update(get_word_set(text))

    def build(self):
        counts = self.counter.most_common()
        max_count = counts[0][1]
        min_count = self.feature_freq * max_count
        words = [word for word, count in counts if count > min_count]
        return TextTransform(self.feature_name, words)


class TextTransform(object):
    def __init__(self, feature_name, words):
        self.feature_name = feature_name
        self.features = [
            ('{}.{}'.format(feature_name, word), word)
            for word in words
        ]

    def get_schema(self):
        return {feature_name: 'bb' for feature_name, _ in self.features}

    def __call__(self, row, transformed_row):
        text = row[self.feature_name]
        word_set = get_word_set(text)
        transformed_row.pop(self.feature_name, None)
        for feature_name, word in self.features:
            transformed_row[feature_name] = (word in word_set)


# ----------------------------------------------------------------------------
# date transforms

EPOCH = dateutil.parser.parse('2014-03-31')  # arbitrary (Loom's birthday)


def days_between(start, end):
    return (end - start).total_seconds() / (24 * 60 * 60)


class DateTransform:
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

    def __call__(self, row, transformed_row):
        feature_name = self.feature_name
        if feature_name in row:
            date = dateutil.parser.parse(row[feature_name])
            del transformed_row[feature_name]

            abs_names = self.abs_names
            transformed_row[abs_names['absolute']] = days_between(EPOCH, date)
            transformed_row[abs_names['mod.year']] = date.month
            transformed_row[abs_names['mod.month']] = date.day
            transformed_row[abs_names['mod.week']] = date.weekday()
            transformed_row[abs_names['mod.day']] = date.hour

            for relative, rel_name in self.rel_names.iteritems():
                if relative in row:
                    other_date = dateutil.parser.parse(row[relative])
                    transformed_row[rel_name] = days_between(other_date, date)


# ----------------------------------------------------------------------------
# commands

def build_transforms(rows_in, builders):
    with loom.util.csv_reader(rows_in) as reader:
        header = reader.next()
        for row in reader:
            row_dict = dict(izip(header, row))
            for builder in builders:
                builder.add_row(row_dict)
    return sum([builder.build() for builder in builders], [])


@loom.documented.transform(
    inputs=['original.schema', 'original.rows_csv'],
    outputs=['ingest.schema', 'ingest.transforms'])
@parsable.command
def make_transforms(schema_in, rows_in, schema_out, transforms_out):
    fluent_schema = json_load(schema_in)
    basic_schema = {}
    transforms = []
    builders = []
    dates = [
        feature_name
        for feature_name, fluent_type in fluent_schema.iteritems()
        if fluent_type.endswith('date')
    ]
    for feature_name, fluent_type in fluent_schema.iteritems():
        if fluent_type.startswith('optional_'):
            fluent_type = fluent_type[len('optional_'):]
            transforms.append(PresenceTransform(feature_name))
        if fluent_type == 'text':
            builders.append(TextTransformBuilder(feature_name))
        elif fluent_type == 'date':
            relatives = [other for other in dates if other < feature_name]
            transforms.append(DateTransform(feature_name, relatives))
        else:
            basic_schema[feature_name] = FLUENT_TYPES[fluent_type]
    if builders:
        transforms += build_transforms(rows_in, builders)
    for transform in transforms:
        basic_schema.update(transform.get_schema())
    json_dump(basic_schema, schema_out)
    pickle_dump(transforms, transforms_out)


@loom.documented.transform(
    inputs=['ingest.schema', 'ingest.transforms', 'original.rows_csv'],
    outputs=['ingest.rows_csv'])
@parsable.command
def transform_rows(schema_in, transforms_in, rows_in, rows_out):
    transforms = pickle_load(transforms_in)
    if not transforms:
        cp_ns(rows_in, rows_out)
    else:
        with ExitStack() as stack:
            with_ = stack.enter_context
            reader = with_(loom.util.csv_reader(rows_in))
            writer = with_(loom.util.csv_writer(rows_out))
            header = reader.next()
            transformed_header = sorted(schema_in.iterkeys())
            writer.writerow(transformed_header)
            for row in reader:
                row_dict = get_row_dict(header, row)
                transformed_row_dict = row_dict.copy()
                for transform in transforms:
                    transform(row_dict, transformed_row_dict)
                writer.writerow(
                    transformed_row_dict.get(key)
                    for key in transformed_header
                )


if __name__ == '__main__':
    parsable.dispatch()
