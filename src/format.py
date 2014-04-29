import os
from itertools import izip
from collections import defaultdict
import ccdb.binary
import loom.schema_pb2
from distributions.io.stream import (
    open_compressed,
    json_load,
    json_dump,
    protobuf_stream_load,
    protobuf_stream_dump,
)
from distributions.dbg.models import dd, dpd, gp, nich
from distributions.lp.clustering import PitmanYor
import parsable
from loom import cFormat
parsable = parsable.Parsable()

try:
    from distributions.dbg.models import bb
    assert bb  # pacify pyflakes
except ImportError:
    bb = None


HASH_FEATURE_MODEL_NAME = {
    'AsymmetricDirichletDiscrete': 0,
    'DPM': 1,
    'GP': 2,
    'NormalInverseChiSq': 3,
}


def hash_feature(feature):
    name = feature['model']
    hash_name = HASH_FEATURE_MODEL_NAME[name]
    params = None

    # DirichletDiscrete features must be ordered by increasing dimension
    if name == 'AsymmetricDirichletDiscrete':
        params = int(feature['parameters']['D'])

    return (hash_name, params)


def get_canonical_feature_ordering(meta):
    features = sorted(
        hash_feature(meta['features'][name]) + (pos, name)
        for pos, name in enumerate(meta['feature_pos'])
    )
    pos_to_name = [feature[-1] for feature in features]
    name_to_pos = {name: pos for pos, name in enumerate(pos_to_name)}
    return {'name_to_pos': name_to_pos, 'pos_to_name': pos_to_name}


def get_short_object_ids(meta):
    #return {_id: i for i, _id in enumerate(sorted(meta['object_pos']))}
    return {_id: i for i, _id in enumerate(meta['object_pos'])}


def get_mixture_filename(dirname, kindid):
    '''
    This must match get_mixture_filename(-,-) in src/cross_cat.cc
    '''
    assert os.path.exists(dirname), 'missing {}'.format(dirname)
    return os.path.join(dirname, 'mixture.{:03d}.pbs.gz'.format(kindid))


def _import_latent_model(meta, ordering, latent, model_out):
    structure = latent['structure']
    get_kindid = {
        feature_name: kindid
        for kindid, kind in enumerate(structure)
        for feature_name in kind['features']
    }
    message = loom.schema_pb2.CrossCatModel()
    kinds = []
    for kind_json in structure:
        kind = message.kinds.add()
        PitmanYor.to_protobuf(
            kind_json['hypers'],
            kind.product_model.clustering.pitman_yor)
        kinds.append(kind)
    for featureid, feature_name in enumerate(ordering['pos_to_name']):
        model_name = meta['features'][feature_name]['model']
        hypers = latent['hypers'][feature_name]
        kindid = get_kindid[feature_name]
        message.featureid_to_kindid.append(kindid)
        kind = kinds[kindid]
        kind.featureids.append(featureid)
        product_model = kind.product_model
        if model_name == 'BetaBernoulli':
            bb.Model.to_protobuf(hypers, product_model.bb.add())
        elif model_name == 'AsymmetricDirichletDiscrete':
            dd.Model.to_protobuf(hypers, product_model.dd.add())
        elif model_name == 'DPM':
            dpd.Model.to_protobuf(hypers, product_model.dpd.add())
        elif model_name == 'GP':
            hypers['inv_beta'] = 1.0 / hypers.pop('beta')
            gp.Model.to_protobuf(hypers, product_model.gp.add())
        elif model_name == 'NormalInverseChiSq':
            nich.Model.to_protobuf(hypers, product_model.nich.add())
        else:
            raise ValueError('unknown model: {}'.format(model_name))
    PitmanYor.to_protobuf(
        latent['model_hypers'],
        message.clustering.pitman_yor)
    with open_compressed(model_out, 'wb') as f:
        f.write(message.SerializeToString())


def _import_latent_groups(meta, ordering, latent, groups_out):
    if not os.path.exists(groups_out):
        os.makedirs(groups_out)

    def groups(kind):
        suffstats = kind['suffstats']
        features = [
            (feature_name, pos, meta['features'][feature_name]['model'])
            for pos, feature_name in enumerate(kind['features'])
        ]
        features.sort(key=(lambda (f, p, m): ordering['name_to_pos'][f]))

        message = loom.schema_pb2.ProductModel.Group()
        for i, category in enumerate(kind['categories']):
            message.count = len(category)
            for feature_name, pos, model_name in features:
                ss = suffstats[pos][i]
                if model_name == 'BetaBernoulli':
                    bb.Model.Group.to_protobuf(ss, message.bb.add())
                elif model_name == 'AsymmetricDirichletDiscrete':
                    dd.Model.Group.to_protobuf(ss, message.dd.add())
                elif model_name == 'DPM':
                    dpd.Model.Group.to_protobuf(ss, message.dpd.add())
                elif model_name == 'GP':
                    ss['count'] = ss.pop('n')
                    gp.Model.Group.to_protobuf(ss, message.gp.add())
                elif model_name == 'NormalInverseChiSq':
                    ss['count_times_variance'] = ss.pop('variance')
                    nich.Model.Group.to_protobuf(ss, message.nich.add())
                else:
                    raise ValueError('unknown model: {}'.format(model_name))
            yield message.SerializeToString()
            message.Clear()

    for kindid, kind in enumerate(latent['structure']):
        filename = get_mixture_filename(groups_out, kindid)
        protobuf_stream_dump(groups(kind), filename)


def _import_latent_assignments(meta, latent, assign_out):
    kind_count = len(latent['structure'])
    groupids_map = defaultdict(lambda: [None] * kind_count)
    for kindid, kind in enumerate(latent['structure']):
        for groupid, cat in enumerate(kind['categories']):
            for long_id in cat:
                groupids_map[long_id][kindid] = groupid
    short_ids = get_short_object_ids(meta)

    def assignments():
        message = loom.schema_pb2.Assignment()
        for long_id, groupids in groupids_map.iteritems():
            message.rowid = short_ids[long_id]
            message.groupids.extend(groupids)
            yield message.SerializeToString()
            message.Clear()

    protobuf_stream_dump(assignments(), assign_out)


@parsable.command
def import_latent(
        meta_in,
        latent_in,
        model_out=None,
        groups_out=None,
        assign_out=None):
    '''
    Import latent from tardis json format.
    '''
    meta = json_load(meta_in)
    ordering = get_canonical_feature_ordering(meta)
    latent = json_load(latent_in)

    if model_out is not None:
        _import_latent_model(meta, ordering, latent, model_out)

    if groups_out is not None:
        _import_latent_groups(meta, ordering, latent, groups_out)

    if assign_out is not None:
        _import_latent_assignments(meta, latent, assign_out)


def _export_latent_model(meta, ordering, model_in):
    message = loom.schema_pb2.CrossCatModel()
    with open_compressed(model_in) as f:
        message.ParseFromString(f.read())

    latent = {
        'hypers': {},
        'structure': [],
        'model_hypers': PitmanYor.from_protobuf(message.clustering.pitman_yor),
        'model_suffstats': {
            'counts': [len(kind.featureids) for kind in message.kinds],
        },
    }
    hypers = latent['hypers']
    structure = latent['structure']
    for kind in message.kinds:
        features = [
            ordering['pos_to_name'][featureid]
            for featureid in kind.featureids
        ]
        product_model = kind.product_model
        structure.append({
            'features': features,
            'categories': [],
            'suffstats': [],
            'hypers': PitmanYor.from_protobuf(
                product_model.clustering.pitman_yor),
            'kind_suffstats': {'counts': []},
        })
        feature_name = iter(features)
        for model in product_model.bb:
            hypers[feature_name.next()] = bb.Model.from_protobuf(model)
        for model in product_model.dd:
            hypers[feature_name.next()] = dd.Model.from_protobuf(model)
        for model in product_model.dpd:
            hypers[feature_name.next()] = dpd.Model.from_protobuf(model)
        for model in product_model.gp:
            hp = gp.Model.from_protobuf(model)
            hp['beta'] = 1.0 / hp.pop('inv_beta')
            hypers[feature_name.next()] = hp
        for model in product_model.nich:
            hypers[feature_name.next()] = nich.Model.from_protobuf(model)

    return latent


def _export_latent_groups(meta, ordering, groups_in, latent):
    assert os.path.exists(groups_in)
    assert os.path.isdir(groups_in)

    def groups(kindid):
        message = loom.schema_pb2.ProductModel.Group()
        filename = get_mixture_filename(groups_in, kindid)
        for string in protobuf_stream_load(filename):
            message.ParseFromString(string)
            yield message
            message.Clear()

    for kindid, kind in enumerate(latent['structure']):
        features = [
            (feature_name, pos, meta['features'][feature_name]['model'])
            for pos, feature_name in enumerate(kind['features'])
        ]
        features.sort(key=(lambda (f, p, m): ordering['name_to_pos'][f]))
        suffstats = [[] for _ in features]
        kind_suffstats = {'counts': []}

        positions = None

        def parse_next(fields, module):
            type_pos = positions[module.__name__]
            positions[module.__name__] += 1
            return module.Model.Group.from_protobuf(fields[type_pos])

        for message in groups(kindid):
            positions = defaultdict(lambda: 0)
            for feature_name, pos, model_name in features:
                if model_name == 'BetaBernoulli':
                    ss = parse_next(message.bb, bb)
                elif model_name == 'AsymmetricDirichletDiscrete':
                    ss = parse_next(message.dd, dd)
                elif model_name == 'DPM':
                    ss = parse_next(message.dpd, dpd)
                elif model_name == 'GP':
                    ss = parse_next(message.gp, gp)
                    ss['n'] = ss.pop('count')
                elif model_name == 'NormalInverseChiSq':
                    ss = parse_next(message.nich, nich)
                    ss['variance'] = ss.pop('count_times_variance')
                else:
                    raise ValueError('unknown model: {}'.format(model_name))
                suffstats[pos].append(ss)
            kind_suffstats['counts'].append(int(message.count))

        kind['suffstats'] = suffstats


def _export_latent_assignments(meta, ordering, assign_in, latent):
    raise NotImplementedError('export assignments')


@parsable.command
def export_latent(
        meta_in,
        model_in,
        latent_out,
        groups_in=None,
        assign_in=None):
    '''
    Export latent to tardis json format.
    '''
    meta = json_load(meta_in)
    ordering = get_canonical_feature_ordering(meta)
    latent = _export_latent_model(meta, ordering, model_in)

    if groups_in is not None:
        _export_latent_groups(meta, ordering, groups_in, latent)

    if assign_in is not None:
        _export_latent_assignments(meta, ordering, assign_in, latent)

    json_dump(latent, latent_out)


def _import_rows(long_ids, short_ids, schema, data, mask):
    message = loom.schema_pb2.SparseRow()
    for long_id, row_data, row_mask in izip(long_ids, data, mask):
        observed = message.data.observed
        message.id = short_ids[long_id]
        for pos, typename, cast in schema:
            if row_mask[pos]:
                observed.append(True)
                fields = getattr(message.data, typename)
                fields.append(cast(row_data[pos]))
            else:
                observed.append(False)
        yield message.SerializeToString()
        message.Clear()


def _cimport_rows(long_ids, short_ids, schema, data, mask):
    message = cFormat.SparseRow()
    add_field = {
        'booeans': message.add_booleans,
        'counts': message.add_counts,
        'reals': message.add_reals,
    }
    for long_id, row_data, row_mask in izip(long_ids, data, mask):
        message.id = short_ids[long_id]
        for pos, typename, cast in schema:
            if row_mask[pos]:
                message.add_observed(True)
                add_field[typename](row_data[pos])
            else:
                message.add_observed(False)
        yield message
        message.Clear()


@parsable.command
def import_data(meta_in, data_in, mask_in, rows_out, validate=False):
    '''
    Import dataset from tardis ccdb binary format.
    '''
    meta = json_load(meta_in)
    long_ids = meta['object_pos']
    features = meta['feature_pos']
    ordering = get_canonical_feature_ordering(meta)
    short_ids = get_short_object_ids(meta)
    get_feature_pos = {name: i for i, name in enumerate(features)}
    schema = []
    for feature_name in ordering['pos_to_name']:
        model_name = meta['features'][feature_name]['model']
        if model_name == 'BetaBernoulli':
            typename = 'booleans'
            cast = bool
        elif model_name in ['AsymmetricDirichletDiscrete', 'DPM', 'GP']:
            typename = 'counts'
            cast = int
        elif model_name == 'NormalInverseChiSq':
            typename = 'reals'
            cast = float
        else:
            raise ValueError('unknown model: {}'.format(model_name))
        schema.append((get_feature_pos[feature_name], typename, cast))
    data, mask = ccdb.binary.load_data(meta, data_in, mask_in, mmap_mode='r')

    if cFormat:
        rows = _cimport_rows(long_ids, short_ids, schema, data, mask)
        cFormat.protobuf_stream_dump(rows, rows_out)
    else:
        rows = _import_rows(long_ids, short_ids, schema, data, mask)
        protobuf_stream_dump(rows, rows_out)

    if validate:
        rows = _import_rows(long_ids, short_ids, schema, data, mask)
        stream = protobuf_stream_load(rows_out)
        for expected, actual in izip(rows, stream):
            assert expected == actual

    if validate and cFormat:
        rows = _cimport_rows(long_ids, short_ids, schema, data, mask)
        stream = cFormat.protobuf_stream_load(rows_out)
        for expected, actual in izip(rows, stream):
            expected = expected.dump()
            actual = actual.dump()
            assert expected == actual, "{} != {}".format(expected, actual)


@parsable.command
def export_data(meta_in, rows_in, rows_out):
    '''
    Export dataset to tarot ccdb json format.
    '''
    raise NotImplementedError('export data')


if __name__ == '__main__':
    parsable.dispatch()
