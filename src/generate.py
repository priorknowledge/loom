import os
import random
import parsable
from distributions.lp.models import bb, dd, dpd, gp, nich
from distributions.lp.clustering import PitmanYor
from distributions.fileutil import tempdir
from distributions.io.stream import open_compressed
import loom.config
import loom.runner
parsable = parsable.Parsable()

CLUSTERING = PitmanYor.from_dict({'alpha': 2.0, 'd': 0.1})
FEATURE_TYPES = {
    'bb': bb,
    'dd': dd,
    'dpd': dpd,
    'gp': gp,
    'nich': nich,
}


def generate_kinds(feature_count):
    '''
    Generate an exponential kind structure, e.g.,
    [o|oo|oooo|oooooooo|oooooooooooooooo|oooooooooooooooooooooooooooooooo]
    '''
    featureid_to_kindid = []
    for i in xrange(feature_count):
        featureid_to_kindid.extend([i] * (2 ** i))
        if len(featureid_to_kindid) >= feature_count:
            break
    featureid_to_kindid = featureid_to_kindid[:feature_count]
    random.shuffle(featureid_to_kindid)
    return featureid_to_kindid


def generate_features(feature_count, feature_type=None):
    get_shared = lambda m: m.Shared.from_dict(m.EXAMPLES[-1]['shared'])
    if feature_type is None:
        features = map(get_shared, FEATURE_TYPES.itervalues())
    else:
        features = [get_shared(FEATURE_TYPES[feature_type])]
    features = features * feature_count
    features = features[:feature_count]
    random.shuffle(features)
    return features


def get_feature_type(feature):
    return feature.__module__.split('.')[-1]


def generate_model(
        row_count,
        feature_count,
        feature_type,
        density):
    features = generate_features(feature_count, feature_type)
    featureid_to_kindid = generate_kinds(feature_count)
    kind_count = 1 + max(featureid_to_kindid)
    cross_cat = loom.schema_pb2.CrossCat()
    kinds = [cross_cat.kinds.add() for _ in xrange(kind_count)]
    for kind in kinds:
        CLUSTERING.dump_protobuf(kind.product_model.clustering.pitman_yor)
    for featureid, feature in enumerate(features):
        kindid = featureid_to_kindid[featureid]
        kind = kinds[kindid]
        feature_type = get_feature_type(feature)
        features = getattr(kind.product_model, feature_type)
        feature.dump_protobuf(features.add())
        kind.featureids.append(featureid)
        cross_cat.featureid_to_kindid.append(kindid)
    CLUSTERING.dump_protobuf(cross_cat.feature_clustering.pitman_yor)
    return cross_cat


@parsable.command
def generate(
        row_count=1000,
        feature_count=100,
        feature_type=None,
        density=0.5,
        model_out='model.pb.gz',
        rows_out='rows.pbs.gz',
        debug=False,
        profile=None):
    '''
    Generate a synthetic dataset.
    '''
    model_out = os.path.abspath(model_out)
    rows_out = os.path.abspath(rows_out)
    root = os.path.abspath(os.path.curdir)

    model = generate_model(row_count, feature_count, feature_type, density)
    with open_compressed(model_out, 'w') as f:
        f.write(model.SerializeToString())

    with tempdir(cleanup_on_error=(not debug)):
        config = {'generate': {'row_count': row_count, 'density': density}}
        config_in = os.path.abspath('config.pb.gz')
        loom.config.config_dump(config, config_in)

        os.chdir(root)
        loom.runner.generate(
            config_in=config_in,
            model_in=model_out,
            rows_out=rows_out,
            debug=debug,
            profile=profile)

        print 'model file is {} bytes'.format(os.path.getsize(model_out))
        print 'rows file is {} bytes'.format(os.path.getsize(rows_out))


if __name__ == '__main__':
    parsable.dispatch()
