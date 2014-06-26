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
import shutil
from itertools import izip
from collections import defaultdict
import csv
import parsable
from distributions.fileutil import tempdir
from distributions.io.stream import open_compressed, json_load, json_dump
import loom.util
import loom.schema
import loom.schema_pb2
import loom.cFormat
parsable = parsable.Parsable()

MAX_CHUNK_COUNT = 1000000

TRUTHY = ['1', '1.0', 'True', 'true', 't']
FALSEY = ['0', '0.0', 'False', 'false', 'f']
BOOLEANS = {
    key: value
    for keys, value in [(TRUTHY, True), (FALSEY, False)]
    for key in keys
}


class DefaultEncoderBuilder(object):
    def __init__(self, encoder):
        self.encoder = encoder

    def add_value(self, value):
        pass

    def __iadd__(self, other):
        pass

    def build(self):
        return self.encoder


class CategoricalEncoderBuilder(object):
    def __init__(self, encoder):
        self.encoder = encoder
        self.counts = defaultdict(lambda: 0)

    def add_value(self, value):
        self.counts[value] += 1

    def __iadd__(self, other):
        for key, value in other.counts.iteritems():
            self.counts[key] += value

    def build(self):
        sorted_keys = [(-count, key) for key, count in self.counts.iteritems()]
        sorted_keys.sort()
        encoder = {key: i for i, (_, key) in enumerate(sorted_keys)}
        self.encoder['encoder'] = encoder
        return self.encoder

    def __getstate__(self):
        return (self.encoder, dict(self.counts))

    def __setstate__(self, (encoder, counts)):
        self.encoder = encoder
        self.counts = defaultdict(lambda: 0)
        self.counts.update(counts)


ENCODER_BUILDERS = defaultdict(lambda: DefaultEncoderBuilder)
ENCODER_BUILDERS['dd'] = CategoricalEncoderBuilder
ENCODER_BUILDERS['dpd'] = CategoricalEncoderBuilder


class CategoricalFakeEncoderBuilder(object):
    def __init__(self, encoder):
        self.encoder = encoder
        self.max_value = -1

    def add_value(self, value):
        self.max_value = max(self.max_value, int(value))

    def build(self):
        encoder = {int(value): value for value in xrange(self.max_value + 1)}
        self.encoder['encoder'] = encoder
        return self.encoder


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
        raise ValueError('unknown model: {}'.format(model_name))


def load_decoder(encoder):
    model_name = encoder['model']
    if model_name in ['dd', 'dpd']:
        decoder = {value: key for key, value in encoder['encoder'].iteritems()}
        return decoder.__getitem__
    elif model_name in ['bb', 'gp', 'nich']:
        return str
    else:
        raise ValueError('unknown model: {}'.format(model_name))


def _make_encoder_builders_file((schema_in, rows_in)):
    assert os.path.isfile(rows_in)
    schema = json_load(schema_in)
    with open_compressed(rows_in, 'rb') as f:
        reader = csv.reader(f)
        header = reader.next()
        builders = []
        for name in header:
            if name in schema:
                model = schema[name]
                encoder = {'name': name, 'model': model}
                Builder = ENCODER_BUILDERS[model]
                builder = Builder(encoder)
            else:
                builder = None
            builders.append(builder)
        for row in reader:
            for value, builder in izip(row, builders):
                if builder is not None:
                    value = value.strip()
                    if value:
                        builder.add_value(value)
    return [b for b in builders if b is not None]


def _make_encoder_builders_dir(schema_in, rows_in):
    assert os.path.isdir(rows_in)
    files_in = [os.path.join(rows_in, f) for f in os.listdir(rows_in)]
    partial_builders = loom.util.parallel_map(_make_encoder_builders_file, [
        (schema_in, file_in)
        for file_in in files_in
    ])
    builders = partial_builders[0]
    for other_builders in partial_builders[1:]:
        for builder, other in izip(builders, other_builders):
            assert builder.encoder == other.encoder
            builder += other
    return builders


def hash_encoder(encoder):
    rank = loom.schema.FEATURE_TYPE_RANK[encoder['model']]
    params = None
    if encoder['model'] == 'dd':
        # dd features must be ordered by increasing dimension
        params = len(encoder['encoder'])
    return (rank, params, encoder['name'])


@parsable.command
def make_encoding(schema_in, rows_in, encoding_out):
    '''
    Make a row encoder from csv rows data + json schema.
    '''
    if os.path.isdir(rows_in):
        builders = _make_encoder_builders_dir(schema_in, rows_in)
    else:
        builders = _make_encoder_builders_file((schema_in, rows_in))
    encoders = [builder.build() for builder in builders]
    encoders.sort(key=hash_encoder)
    json_dump(encoders, encoding_out)


@parsable.command
def make_fake_encoding(model_in, rows_in, schema_out, encoding_out):
    '''
    Make a fake encoding from protobuf formatted model + rows.
    '''
    cross_cat = loom.schema_pb2.CrossCat()
    with open_compressed(model_in, 'rb') as f:
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
    builders = [FAKE_ENCODER_BUILDERS[e['model']](e) for e in encoders]
    for row in loom.cFormat.row_stream_load(rows_in):
        data = row.iter_data()
        observeds = data['observed']
        for observed, field, builder in izip(observeds, fields, builders):
            if observed:
                builder.add_value(str(data[field].next()))
    for builder in builders:
        builder.build()
    json_dump(encoders, encoding_out)


def _import_rows_file((encoding_in, rows_in, rows_out, id_offset, id_stride)):
    assert os.path.isfile(rows_in)
    encoders = json_load(encoding_in)
    message = loom.cFormat.Row()
    add_field = {
        'booleans': message.add_booleans,
        'counts': message.add_counts,
        'reals': message.add_reals,
    }
    with open_compressed(rows_in, 'rb') as f:
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
                message.id = id_offset + id_stride * i
                for pos, add, encode in schema:
                    value = None if pos is None else row[pos].strip()
                    observed = bool(value)
                    message.add_observed(observed)
                    if observed:
                        add(encode(value))
                yield message
                message.Clear()

        loom.cFormat.row_stream_dump(rows(), rows_out)


def _import_rows_dir(encoding_in, rows_in, rows_out, id_offset, id_stride):
    assert os.path.isdir(rows_in)
    files_in = sorted(
        os.path.abspath(os.path.join(rows_in, f))
        for f in os.listdir(rows_in)
    )
    file_count = len(files_in)
    assert file_count > 0, 'no files in {}'.format(rows_in)
    assert file_count < 1e6, 'too many files in {}'.format(rows_in)
    files_out = []
    tasks = []
    for i, file_in in enumerate(files_in):
        file_out = 'part_{:06d}.{}'.format(i, os.path.basename(rows_out))
        offset = id_offset + id_stride * i
        stride = id_stride * file_count
        files_out.append(file_out)
        tasks.append((encoding_in, file_in, file_out, offset, stride))
    rows_out = os.path.abspath(rows_out)
    with tempdir():
        loom.util.parallel_map(_import_rows_file, tasks)
        # It is safe use open instead of open_compressed even for .gz files;
        # see http://stackoverflow.com/questions/8005114
        with open(rows_out, 'wb') as whole:
            for file_out in files_out:
                with open(file_out, 'rb') as part:
                    shutil.copyfileobj(part, whole)
                os.remove(file_out)


@parsable.command
def import_rows(encoding_in, rows_in, rows_out):
    '''
    Import rows from csv format to protobuf-stream format.
    rows_in can be a csv file or a directory containing csv files.
    Any csv file may be be raw .csv, or compressed .csv.gz or .csv.bz2.
    '''
    id_offset = 0
    id_stride = 1
    args = (encoding_in, rows_in, rows_out, id_offset, id_stride)
    if os.path.isdir(rows_in):
        _import_rows_dir(*args)
    else:
        _import_rows_file(args)


@parsable.command
def export_rows(encoding_in, rows_in, rows_out, chunk_size=1000000):
    '''
    Export rows from protobuf stream to csv.
    '''
    for ext in ['.csv', '.gz', '.bz2']:
        assert not rows_out.endswith(ext), 'rows_out should be a dirname'
    assert chunk_size > 0
    encoders = json_load(encoding_in)
    fields = [loom.schema.MODEL_TO_DATATYPE[e['model']] for e in encoders]
    decoders = [load_decoder(e) for e in encoders]
    header = [e['name'] for e in encoders]
    if os.path.exists(rows_out):
        shutil.rmtree(rows_out)
    os.makedirs(rows_out)
    rows = loom.cFormat.row_stream_load(rows_in)
    try:
        empty = None
        for i in xrange(MAX_CHUNK_COUNT):
            file_out = os.path.join(rows_out, 'rows_{:06d}.csv'.format(i))
            with open_compressed(file_out, 'wb') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                empty = file_out
                for j in xrange(chunk_size):
                    data = rows.next().iter_data()
                    schema = izip(data['observed'], fields, decoders)
                    row = [
                        decode(data[field].next()) if observed else ''
                        for observed, field, decode in schema
                    ]
                    writer.writerow(row)
                    empty = None
    except StopIteration:
        if empty:
            os.remove(empty)


if __name__ == '__main__':
    parsable.dispatch()
