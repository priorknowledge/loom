from distributions.lp.models import bb, dd, dpd, gp, bnb, nich

FEATURES = [bb, dd, dpd, gp, bnb, nich]

FEATURES = [bb, dd, dpd, gp, nich]

FEATURE_TYPES = {
    module.__name__.split('.')[-1]: module
    for module in FEATURES
}

FEATURE_TYPE_RANK = {
    module.__name__.split('.')[-1]: i
    for i, module in enumerate(FEATURES)
}


def get_feature_type(feature):
    return feature.__module__.split('.')[-1]


def get_feature_rank(shared):
    feature_type = get_feature_type(shared)
    if feature_type == 'dd':
        param = len(shared.dump()['alphas'])
    else:
        param = None
    return (FEATURE_TYPE_RANK[feature_type], param)


def get_canonical_feature_ordering(named_features):
    features = sorted(
        (get_feature_rank(feature), name)
        for name, feature in named_features.iteritems()
    )
    pos_to_name = [name for _, name in features]
    name_to_pos = {name: pos for pos, name in enumerate(pos_to_name)}
    return {'pos_to_name': pos_to_name, 'name_to_pos': name_to_pos}


def sort_features(features):
    features.sort(key=get_feature_rank)
