import loom.schema_pb2
from distributions.fileutil import json_load, json_dump
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
        params = int(feature['params']['D'])

    return (hash_name, params)


def get_canonical_ordering(meta):
    features = sorted(
        hash_feature(meta['features'][name]) + (pos, name)
        for pos, name in enumerate(meta['featurepos'])
    )
    decode = [feature[-1] for feature in features]
    encode = {name: pos for pos, name in enumerate(decode)}
    return {'encode': encode, 'decode': decode}


def _import_model(Model, json, message):
    Model.model_load(json).dump_protobuf(message)


def _export_model(Model, message):
    model = Model()
    model.load_protobuf(message)
    return model.dump()


@parsable.command
def import_latent(
        meta_in,
        latent_in,
        model_out,
        groups_out=None,
        assignments_out=None):
    '''
    Import latent from tardis json format.
    '''
    meta = json_load(meta_in)
    ordering = get_canonical_ordering(meta)

    latent = json_load(latent_in)
    kinds = latent['structure']
    get_kindid = {
        feature_name: kindid
        for kindid, kind in enumerate(kinds)
        for feature_name in kind['features']
    }

    cross_cat_model = loom.schema_pb2.CrossCatModel()
    kinds = [cross_cat_model.add() for _ in xrange(len(kinds))]
    for featureid, feature_name in enumerate(ordering['decode']):
        model_name = meta['features'][feature_name]['model']
        hypers = latent['hypers'][feature_name]
        kindid = get_kindid[feature_name]
        cross_cat_model.featureid_to_kindid.append(kindid)
        kind = kinds[kindid]
        kind.featureids.append(featureid)
        product_model = kind.product_model
        _import_model(
            PitmanYor,
            kind['hypers'],
            product_model.clustering.pitman_yor)
        if model_name == 'AsymmetricDirichletDiscrete':
            _import_model(dd.Model, hypers, product_model.dd.add())
        elif model_name == 'DPM':
            _import_model(dpd.Model, hypers, product_model.dpd.add())
        elif model_name == 'GP':
            _import_model(gp.Model, hypers, product_model.gp.add())
        elif model_name == 'NormalInverseChiSq':
            _import_model(nich.Model, hypers, product_model.nich.add())
        else:
            raise ValueError('unknown model name: {}'.format(model_name))
    _import_model(
        PitmanYor,
        latent['model_hypers'],
        cross_cat_model.clustering.pitman_yor)
    with open(model_out, 'wb') as f:
        f.write(cross_cat_model.SerializeToString())

    if groups_out is not None:
        raise NotImplementedError('dump groups')

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
    ordering = get_canonical_ordering(meta)

    cross_cat_model = loom.schema_pb2.CrossCatModel()
    with open(model_in) as f:
        cross_cat_model.ParseFromString(f.read())

    latent = {
        'hypers': {},
        'structure': [],
        'model_hypers': _export_model(
            PitmanYor,
            cross_cat_model.clustering.pitman_yor)
    }
    hypers = latent['hypers']
    structure = latent['structure']
    for kind in cross_cat_model.kinds:
        features = [ordering[featureid] for featureid in kind.featureids]
        product_model = kind.product_model
        structure.append({
            'features': features,
            'categories': [],
            'suffstats': [],
            'hypers': _export_model(
                PitmanYor,
                product_model.clustering.pitman_yor),
        })
        feature_name = iter(features)
        for model in product_model.dd:
            hypers[feature_name.next()] = _export_model(dd.Model, model)
        for model in product_model.dpd:
            hypers[feature_name.next()] = _export_model(dpd.Model, model)
        for model in product_model.gp:
            hypers[feature_name.next()] = _export_model(gp.Model, model)
        for model in product_model.nich:
            hypers[feature_name.next()] = _export_model(nich.Model, model)

    if groups_in is not None:
        raise NotImplementedError('export groups')

    if assignments_in is not None:
        raise NotImplementedError('export assignments')

    json_dump(latent, latent_out)


@parsable.command
def import_data(
        meta_in,
        data_in,
        mask_in,
        values_out):
    '''
    Import dataset from tardis ccdb binary format.
    '''
    raise NotImplementedError('import data')


@parsable.command
def export_data(
        meta_in,
        values_in,
        rows_out):
    '''
    Export dataset to tarot ccdb json format.
    '''
    raise NotImplementedError('export data')


if __name__ == '__main__':
    parsable.dispatch()
