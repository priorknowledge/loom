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
from collections import defaultdict
import csv
import parsable
from distributions.io.stream import open_compressed, json_load, json_dump
import loom.schema
import loom.schema_pb2
import loom.cFormat
parsable = parsable.Parsable()

TRUTHY = ['1', '1.0', 'True', 'true']
FALSEY = ['0', '0.0', 'False', 'false']
BOOLEANS = {
    key: value
    for keys, value in [(TRUTHY, True), (FALSEY, False)]
    for key in keys
}


class CategoricalEncoderBuilder:
    def __init__(self):
        self.counts = defaultdict(lambda: 0)

    def add_value(self, value):
        self.counts[value] += 1

    def dump(self):
        sorted_keys = [(-count, key) for key, count in self.counts.iteritems()]
        sorted_keys.sort()
        return {key: i for i, (_, key) in enumerate(sorted_keys)}


class DefaultEncoderBuilder:
    def add_value(self, value):
        pass

    def dump(self):
        return None


ENCODER_BUILDERS = defaultdict(lambda: DefaultEncoderBuilder)
ENCODER_BUILDERS['dd'] = CategoricalEncoderBuilder
ENCODER_BUILDERS['dpd'] = CategoricalEncoderBuilder


class CategoricalFakeEncoderBuilder:
    def __init__(self):
        self.max_value = -1

    def add_value(self, value):
        self.max_value = max(self.max_value, int(value))

    def dump(self):
        return {int(value): value for value in xrange(self.max_value + 1)}


FAKE_ENCODER_BUILDERS = defaultdict(lambda: DefaultEncoderBuilder)
FAKE_ENCODER_BUILDERS['dd'] = CategoricalFakeEncoderBuilder
FAKE_ENCODER_BUILDERS['dpd'] = CategoricalFakeEncoderBuilder


def load_encoder(encoder):
    model_name = encoder['model']
    if model_name == 'bb':
        return BOOLEANS.__getitem__
    elif model_name in ['dd', 'dpd']:
        return encoder['encoder'].__getitem__
    elif model_name == 'gp':
        return int
    elif model_name == 'nich':
        return float
    else:
        raise ValueError('unknown model name: {}'.format(model_name))


def load_decoder(encoder):
    model_name = encoder['model']
    if model_name in ['dd', 'dpd']:
        #decoder = [None] * len(encoder['encoder'])
        #for key, value in encoder['encoder'].iteritems():
        #    decoder[int(value)] = key
        decoder = {value: key for key, value in encoder['encoder'].iteritems()}
        return decoder.__getitem__
    elif model_name in ['bb', 'gp', 'nich']:
        return str
    else:
        raise ValueError('unknown model name: {}'.format(model_name))


def hash_encoder(encoder):
    rank = loom.schema.FEATURE_TYPE_RANK[encoder['model']]
    params = None

    # DirichletDiscrete features must be ordered by increasing dimension
    if encoder['model'] == 'dd':
        params = len(encoder['encoder'])
    return (rank, params)


@parsable.command
def make_encoding(schema_in, rows_in, encoding_out):
    '''
    Make a row encoder from csv rows data + json schema.
    '''
    schema = json_load(schema_in)
    with open_compressed(rows_in) as f:
        reader = csv.reader(f)
        feature_names = list(reader.next())
        encoders = [
            {'name': name, 'model': schema[name]}
            for name in feature_names
            if name in schema
        ]
        builders = [ENCODER_BUILDERS[e['model']]() for e in encoders]
        for row in reader:
            for value, builder in izip(row, builders):
                value = value.strip()
                if value:
                    builder.add_value(value)
    for encoder, builder in izip(encoders, builders):
        encoder['encoder'] = builder.dump()
    encoders.sort(key=hash_encoder)
    json_dump(encoders, encoding_out)


@parsable.command
def make_fake_encoding(model_in, rows_in, schema_out, encoding_out):
    '''
    Make a fake encoding from protobuf formatted model + rows.
    '''
    cross_cat = loom.schema_pb2.CrossCat()
    with open_compressed(model_in) as f:
        cross_cat.ParseFromString(f.read())
    schema = {}
    for kind in cross_cat.kinds:
        featureid = iter(kind.featureids)
        for model in loom.schema.FEATURES:
            model_name = model.__name__.split('.')[-1]
            for shared in getattr(kind.product_model, model_name):
                feature_name = '{:06d}'.format(featureid.next())
                schema[feature_name] = model_name
    json_dump(schema, schema_out)
    encoders = [{'name': k, 'model': v} for k, v in sorted(schema.iteritems())]
    fields = [loom.schema.MODEL_TO_DATATYPE[e['model']] for e in encoders]
    builders = [FAKE_ENCODER_BUILDERS[e['model']]() for e in encoders]
    for row in loom.cFormat.row_stream_load(rows_in):
        data = row.iter_data()
        observeds = data['observed']
        for observed, field, builder in izip(observeds, fields, builders):
            if observed:
                builder.add_value(str(data[field].next()))
    for encoder, builder in izip(encoders, builders):
        encoder['encoder'] = builder.dump()
    json_dump(encoders, encoding_out)


@parsable.command
def import_rows(encoding_in, rows_in, rows_out):
    '''
    Import rows from csv rows to protobuf stream rows.
    '''
    encoders = json_load(encoding_in)
    message = loom.cFormat.Row()
    add_field = {
        'booleans': message.add_booleans,
        'counts': message.add_counts,
        'reals': message.add_reals,
    }
    with open_compressed(rows_in) as f:
        reader = csv.reader(f)
        feature_names = list(reader.next())
        name_to_pos = {name: i for i, name in enumerate(feature_names)}
        schema = []
        for encoder in encoders:
            pos = name_to_pos.get(encoder['name'])
            add = add_field[loom.schema.MODEL_TO_DATATYPE[encoder['model']]]
            encode = load_encoder(encoder)
            schema.append((pos, add, encode))

        def rows():
            for i, row in enumerate(reader):
                message.id = i
                for pos, add, encode in schema:
                    value = None if pos is None else row[pos].strip()
                    observed = bool(value)
                    message.add_observed(observed)
                    if observed:
                        add(encode(value))
                yield message
                message.Clear()

        loom.cFormat.row_stream_dump(rows(), rows_out)


@parsable.command
def export_rows(encoding_in, rows_in, rows_out):
    '''
    Export rows from protobuf stream to csv.
    '''
    encoders = json_load(encoding_in)
    fields = [loom.schema.MODEL_TO_DATATYPE[e['model']] for e in encoders]
    decoders = [load_decoder(e) for e in encoders]
    with open_compressed(rows_out, 'wb') as f:
        writer = csv.writer(f)
        writer.writerow([e['name'] for e in encoders])
        for message in loom.cFormat.row_stream_load(rows_in):
            data = message.iter_data()
            schema = izip(data['observed'], fields, decoders)
            row = [
                decode(data[field].next()) if observed else ''
                for observed, field, decode in schema
            ]
            writer.writerow(row)


if __name__ == '__main__':
    parsable.dispatch()
