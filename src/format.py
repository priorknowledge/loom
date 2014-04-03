import loom.schema_pb2
from distributions.fileutil import json_load, json_dump
from distributions.dbg.models import dd, dpd, gp, nich
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


def _import_model(module, hypers, messages):
    module.Model.model_load(hypers).dump_protobuf(messages.add())


def _export_model(module, message):
    model = module.Model()
    model.load_protobuf(message)
    hypers = model.dump()
    return hypers


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
        if model_name == 'AsymmetricDirichletDiscrete':
            _import_model(dd, hypers, product_model.dd)
        elif model_name == 'DPM':
            _import_model(dpd, hypers, product_model.dpd)
        elif model_name == 'GP':
            _import_model(gp, hypers, product_model.gp)
        elif model_name == 'NormalInverseChiSq':
            _import_model(nich, hypers, product_model.nich)
        else:
            raise ValueError('unknown model name: {}'.format(model_name))
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
    }
    hypers = latent['hypers']
    structure = latent['structure']
    for kind in cross_cat_model.kinds:
        featureids = kind.featureids
        structure.append({
            'features': [ordering[featureid] for featureid in featureids],
            'categories': [],
            'suffstats': [],
        })
        product_model = kind.product_model
        featureid = iter(featureids)
        for model in product_model.dd:
            hypers[ordering[featureid.next()]] = _export_model(dd, model)
        for model in product_model.dpd:
            hypers[ordering[featureid.next()]] = _export_model(dpd, model)
        for model in product_model.gp:
            hypers[ordering[featureid.next()]] = _export_model(gp, model)
        for model in product_model.nich:
            hypers[ordering[featureid.next()]] = _export_model(nich, model)

    if groups_in is not None:
        raise NotImplementedError('TODO')

    if assignments_in is not None:
        raise NotImplementedError('TODO')

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
    raise NotImplementedError('TODO')


@parsable.command
def export_data(
        meta_in,
        values_in,
        rows_out):
    '''
    Export dataset to tarot ccdb json format.
    '''
    raise NotImplementedError('TODO')


if __name__ == '__main__':
    parsable.dispatch()
