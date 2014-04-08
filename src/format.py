import os
from itertools import izip
import ccdb.binary
import loom.schema_pb2
from distributions.io.stream import (
    open_compressed,
    json_load,
    json_dump,
    protobuf_stream_dump,
)
from distributions.dbg.models import dd, dpd, gp, nich
from distributions.lp.clustering import PitmanYor
import parsable
parsable = parsable.Parsable()


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
    return {_id: i for i, _id in enumerate(sorted(meta['object_pos']))}


def get_mixture_filename(dirname, kindid):
    '''
    This must match get_mixture_filename(-,-) in src/cross_cat.cc
    '''
    assert os.path.exists(dirname), 'missing {}'.format(dirname)
    return os.path.join(dirname, 'mixture.{:03d}.pbs.gz'.format(kindid))


def json_to_pb(Model, json, message):
    model = Model()
    model.load(json)
    model.dump_protobuf(message)


def pb_to_json(Model, message):
    model = Model()
    model.load_protobuf(message)
    return model.dump()


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
        json_to_pb(
            PitmanYor,
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
        if model_name == 'AsymmetricDirichletDiscrete':
            json_to_pb(dd.Model, hypers, product_model.dd.add())
        elif model_name == 'DPM':
            json_to_pb(dpd.Model, hypers, product_model.dpd.add())
        elif model_name == 'GP':
            hypers['inv_beta'] = 1.0 / hypers.pop('beta')
            json_to_pb(gp.Model, hypers, product_model.gp.add())
        elif model_name == 'NormalInverseChiSq':
            json_to_pb(nich.Model, hypers, product_model.nich.add())
        else:
            raise ValueError('unknown model: {}'.format(model_name))
    json_to_pb(
        PitmanYor,
        latent['model_hypers'],
        message.clustering.pitman_yor)
    with open_compressed(model_out, 'wb') as f:
        f.write(message.SerializeToString())


def _import_latent_groups(meta, ordering, latent, groups_out):
    if not os.path.exists(groups_out):
        os.makedirs(groups_out)
    message = loom.schema_pb2.ProductModel.Group()
    for kindid, kind in enumerate(latent['structure']):
        suffstats = kind['suffstats']
        features = [
            (feature_name, pos, meta['features'][feature_name]['model'])
            for pos, feature_name in enumerate(kind['features'])
        ]
        features.sort(key=(lambda (f, p, m): ordering['name_to_pos'][f]))

        def groups():
            for i, category in enumerate(kind['categories']):
                message.count = len(category)
                for feature_name, pos, model_name in features:
                    ss = suffstats[pos][i]
                    if model_name == 'AsymmetricDirichletDiscrete':
                        print 'DEBUG', ss['counts']
                        json_to_pb(dd.Model.Group, ss, message.dd.add())
                    elif model_name == 'DPM':
                        json_to_pb(dpd.Model.Group, ss, message.dpd.add())
                    elif model_name == 'GP':
                        ss['count'] = ss.pop('n')
                        json_to_pb(gp.Model.Group, ss, message.gp.add())
                    elif model_name == 'NormalInverseChiSq':
                        ss['count_times_variance'] = ss.pop('variance')
                        json_to_pb(nich.Model.Group, ss, message.nich.add())
                    else:
                        raise ValueError(
                            'unknown model: {}'.format(model_name))
                yield message.SerializeToString()
                message.Clear()

        filename = get_mixture_filename(groups_out, kindid)
        protobuf_stream_dump(groups(), filename)


@parsable.command
def import_latent(
        meta_in,
        latent_in,
        model_out=None,
        groups_out=None,
        assignments_out=None):
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

    if assignments_out is not None:
        raise NotImplementedError('dump assignments')


@parsable.command
def export_latent(
        meta_in,
        model_in,
        latent_out,
        groups_in=None,
        assignments_in=None):
    '''
    Export latent to tardis json format.
    '''
    meta = json_load(meta_in)
    ordering = get_canonical_feature_ordering(meta)

    message = loom.schema_pb2.CrossCatModel()
    with open_compressed(model_in) as f:
        message.ParseFromString(f.read())

    latent = {
        'hypers': {},
        'structure': [],
        'model_hypers': pb_to_json(PitmanYor, message.clustering.pitman_yor)
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
            'hypers': pb_to_json(
                PitmanYor,
                product_model.clustering.pitman_yor),
        })
        feature_name = iter(features)
        for model in product_model.dd:
            hypers[feature_name.next()] = pb_to_json(dd.Model, model)
        for model in product_model.dpd:
            hypers[feature_name.next()] = pb_to_json(dpd.Model, model)
        for model in product_model.gp:
            hp = pb_to_json(gp.Model, model)
            hp['beta'] = 1.0 / hp.pop('inv_beta')
            hypers[feature_name.next()] = hp
        for model in product_model.nich:
            hypers[feature_name.next()] = pb_to_json(nich.Model, model)

    if groups_in is not None:
        raise NotImplementedError('export groups')

    if assignments_in is not None:
        raise NotImplementedError('export assignments')

    json_dump(latent, latent_out)


@parsable.command
def import_data(meta_in, data_in, mask_in, values_out):
    '''
    Import dataset from tardis ccdb binary format.
    '''
    meta = json_load(meta_in)
    objects = meta['object_pos']
    features = meta['feature_pos']
    ordering = get_canonical_feature_ordering(meta)
    short_ids = get_short_object_ids(meta)
    get_feature_pos = {name: i for i, name in enumerate(features)}
    row = loom.schema_pb2.SparseRow()
    schema = []
    for feature_name in ordering['pos_to_name']:
        model_name = meta['features'][feature_name]['model']
        if model_name in ['AsymmetricDirichletDiscrete', 'DPM', 'GP']:
            typename = 'counts'
            cast = int
        elif model_name == 'NormalInverseChiSq':
            typename = 'reals'
            cast = float
        else:
            raise ValueError('unknown model: {}'.format(model_name))
        schema.append((get_feature_pos[feature_name], typename, cast))
    data, mask = ccdb.binary.load_data(meta, data_in, mask_in, mmap_mode='r')

    def rows():
        for long_id, row_data, row_mask in izip(objects, data, mask):
            observed = row.data.observed
            row.id = short_ids[long_id]
            for pos, typename, cast in schema:
                if row_mask[pos]:
                    observed.append(True)
                    getattr(row.data, typename).append(cast(row_data[pos]))
                else:
                    observed.append(False)
            yield row.SerializeToString()
            row.data.Clear()

    protobuf_stream_dump(rows(), values_out)


@parsable.command
def export_data(meta_in, values_in, rows_out):
    '''
    Export dataset to tarot ccdb json format.
    '''
    raise NotImplementedError('export data')


if __name__ == '__main__':
    parsable.dispatch()
