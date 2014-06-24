from itertools import izip
from collections import defaultdict
import csv
import numpy
from distributions.io.stream import (
    open_compressed,
    json_load,
    json_dump,
)
from distributions.lp.clustering import PitmanYor
import loom.schema
import loom.hyperprior
import loom.cFormat
import parsable
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


ENCODER_BUILDERS = {
    'bb': DefaultEncoderBuilder,
    'dd': CategoricalEncoderBuilder,
    'dpd': CategoricalEncoderBuilder,
    'gp': DefaultEncoderBuilder,
    'nich': DefaultEncoderBuilder,
}


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
        encoder['encoder'] = builders.dump()
    encoders.sort(key=hash_encoder)
    json_dump(encoders, encoding_out)


def load_encoder(encoding):
    model_name = encoding['model']
    if model_name == 'bb':
        return BOOLEANS.__getitem__
    elif model_name in ['dd', 'dpd']:
        return encoding['encoder'].__getitem__
    elif model_name == 'gp':
        return int
    elif model_name == 'nich':
        return float
    else:
        raise ValueError('unknown model name: {}'.format(model_name))


@parsable.command
def import_rows(encoding_in, rows_in, rows_out):
    '''
    Import csv rows into protobuf stream rows.
    '''
    encoders = json_load(encoding_in)
    message = loom.cFormat.Row()
    add_field = {
        'booeans': message.add_booleans,
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
            cast = load_encoder(encoder)
            schema.append((pos, add, cast))

        def rows():
            for row in reader:
                for pos, add, cast in schema:
                    value = None if pos is None else row[pos].strip()
                    observed = bool(value)
                    row.add_observed(observed)
                    if observed:
                        add(cast(value))
                yield message
                message.Clear()

        loom.cFormat.row_stream_dump(rows(), rows_out)


@parsable.command
def init_model(encoding_in, model_out, seed=0):
    '''
    Create initial model with a single kind.
    '''
    encoding = json_load(encoding_in)
    numpy.random.seed(seed)
    cross_cat = loom.schema_pb2.CrossCat()
    kind = cross_cat.kinds.add()
    kind.featureids.extend(range(len(encoding)))
    raise NotImplementedError('TODO generate random shareds')
    clustering = PitmanYor.from_dict(loom.hyperprior.sample('clustering'))
    clustering.dump(kind.clustering)
    topology = PitmanYor.from_dict(loom.hyperprior.sample('topology'))
    topology.dump(cross_cat.topology)
    with open_compressed(model_out, 'w') as f:
        f.write(cross_cat.SerializeToString())


if __name__ == '__main__':
    parsable.dispatch()
