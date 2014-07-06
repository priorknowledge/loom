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
BOOLEAN_SYMBOLS = {
    key: value
    for keys, value in [(TRUTHY, True), (FALSEY, False)]
    for key in keys
}


class DefaultEncoderBuilder(object):
    def __init__(self, name, model):
        self.name = name
        self.model = model

    def add_value(self, value):
        pass

    def __iadd__(self, other):
        pass

    def build(self):
        return {
            'name': self.name,
            'model': self.model,
        }


class CategoricalEncoderBuilder(object):
    def __init__(self, name, model):
        self.name = name
        self.model = model
        self.counts = defaultdict(lambda: 0)

    def add_value(self, value):
        self.counts[value] += 1

    def __iadd__(self, other):
        for key, value in other.counts.iteritems():
            self.counts[key] += value

    def build(self):
        sorted_keys = [(-count, key) for key, count in self.counts.iteritems()]
        sorted_keys.sort()
        symbols = {key: i for i, (_, key) in enumerate(sorted_keys)}
        return {
            'name': self.name,
            'model': self.model,
            'symbols': symbols,
        }

    def __getstate__(self):
        return (self.name, self.model, dict(self.counts))

    def __setstate__(self, (name, model, counts)):
        self.name = name
        self.model = model
        self.counts = defaultdict(lambda: 0)
        self.counts.update(counts)


ENCODER_BUILDERS = defaultdict(lambda: DefaultEncoderBuilder)
ENCODER_BUILDERS['dd'] = CategoricalEncoderBuilder
ENCODER_BUILDERS['dpd'] = CategoricalEncoderBuilder


class CategoricalFakeEncoderBuilder(object):
    def __init__(self, name, model):
        self.name = name
        self.model = model
        self.max_value = -1

    def add_value(self, value):
        self.max_value = max(self.max_value, int(value))

    def build(self):
        symbols = {int(value): value for value in xrange(self.max_value + 1)}
        return {
            'name': self.name,
            'model': self.model,
            'symbols': symbols,
        }


FAKE_ENCODER_BUILDERS = defaultdict(lambda: DefaultEncoderBuilder)
FAKE_ENCODER_BUILDERS['dd'] = CategoricalFakeEncoderBuilder
FAKE_ENCODER_BUILDERS['dpd'] = CategoricalFakeEncoderBuilder


def load_encoder(encoder):
    model = encoder['model']
    if model == 'bb':
        encode = BOOLEAN_SYMBOLS.__getitem__
    elif model in ['dd', 'dpd']:
        encode = encoder['symbols'].__getitem__
    else:
        encode = loom.schema.FEATURE_TYPES[model].Value
    return encode


def load_decoder(encoder):
    model = encoder['model']
    if model == 'bb':
        decode = ('0', '1').__getitem__
    elif model in ['dd', 'dpd']:
        decoder = {value: key for key, value in encoder['symbols'].iteritems()}
        decode = decoder.__getitem__
    else:
        decode = str
    return decode


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
                Builder = ENCODER_BUILDERS[model]
                builder = Builder(name, model)
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
            assert builder.name == other.name
            builder += other
    return builders


def get_encoder_rank(encoder):
    rank = loom.schema.FEATURE_TYPE_RANK[encoder['model']]
    params = None
    if encoder['model'] == 'dd':
        # dd features must be ordered by increasing dimension
        params = len(encoder['symbols'])
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
    encoders.sort(key=get_encoder_rank)
    json_dump(encoders, encoding_out)


def ensure_fake_encoders_are_sorted(encoders):
    dds = [e['symbols'] for e in encoders if e['model'] == 'dd']
    for smaller, larger in izip(dds, dds[1:]):
        if len(smaller) > len(larger):
            larger.update(smaller)


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
        for module in loom.schema.FEATURES:
            model = module.__name__.split('.')[-1]
            for shared in getattr(kind.product_model, model):
                feature_name = '{:06d}'.format(featureid.next())
                schema[feature_name] = model
    json_dump(schema, schema_out)
    fields = []
    builders = []
    for name, model in sorted(schema.iteritems()):
        fields.append(loom.schema.MODEL_TO_DATATYPE[model])
        Builder = FAKE_ENCODER_BUILDERS[model]
        builder = Builder(name, model)
        builders.append(builder)
    for row in loom.cFormat.row_stream_load(rows_in):
        data = row.iter_data()
        observeds = data['observed']
        for observed, field, builder in izip(observeds, fields, builders):
            if observed:
                builder.add_value(str(data[field].next()))
    encoders = [builder.build() for builder in builders]
    ensure_fake_encoders_are_sorted(encoders)
    json_dump(encoders, encoding_out)


def _import_rows_file(args):
    encoding_in, rows_csv_in, rows_out, id_offset, id_stride = args
    assert os.path.isfile(rows_csv_in)
    encoders = json_load(encoding_in)
    message = loom.cFormat.Row()
    add_field = {
        'booleans': message.add_booleans,
        'counts': message.add_counts,
        'reals': message.add_reals,
    }
    with open_compressed(rows_csv_in, 'rb') as f:
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


def _import_rows_dir(encoding_in, rows_csv_in, rows_out, id_offset, id_stride):
    assert os.path.isdir(rows_csv_in)
    files_in = sorted(
        os.path.abspath(os.path.join(rows_csv_in, f))
        for f in os.listdir(rows_csv_in)
    )
    file_count = len(files_in)
    assert file_count > 0, 'no files in {}'.format(rows_csv_in)
    assert file_count < 1e6, 'too many files in {}'.format(rows_csv_in)
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
def import_rows(encoding_in, rows_csv_in, rows_out):
    '''
    Import rows from csv format to protobuf-stream format.
    rows_csv_in can be a csv file or a directory containing csv files.
    Any csv file may be be raw .csv, or compressed .csv.gz or .csv.bz2.
    '''
    id_offset = 0
    id_stride = 1
    args = (encoding_in, rows_csv_in, rows_out, id_offset, id_stride)
    if os.path.isdir(rows_csv_in):
        _import_rows_dir(*args)
    else:
        _import_rows_file(args)


@parsable.command
def export_rows(encoding_in, rows_in, rows_csv_out, chunk_size=1000000):
    '''
    Export rows from protobuf stream to csv.
    '''
    for ext in ['.csv', '.gz', '.bz2']:
        assert not rows_csv_out.endswith(ext),\
            'rows_csv_out should be a dirname'
    assert chunk_size > 0
    encoders = json_load(encoding_in)
    fields = [loom.schema.MODEL_TO_DATATYPE[e['model']] for e in encoders]
    decoders = [load_decoder(e) for e in encoders]
    header = [e['name'] for e in encoders]
    if os.path.exists(rows_csv_out):
        shutil.rmtree(rows_csv_out)
    os.makedirs(rows_csv_out)
    rows = loom.cFormat.row_stream_load(rows_in)
    try:
        empty = None
        for i in xrange(MAX_CHUNK_COUNT):
            file_out = os.path.join(
                rows_csv_out,
                'rows_{:06d}.csv.gz'.format(i))
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
