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
import numpy.random
import parsable
from distributions.lp.clustering import PitmanYor
from loom.util import tempdir
from distributions.io.stream import open_compressed, json_load
import loom.schema
import loom.hyperprior
import loom.config
import loom.runner
parsable = parsable.Parsable()

CLUSTERING = PitmanYor.from_dict({'alpha': 2.0, 'd': 0.1})


def random_choice(grid):
    try:
        return numpy.random.choice(grid)
    except AttributeError:
        return grid[numpy.random.randint(len(grid))]


def sample_grid(grid):
    if isinstance(grid, list):
        return random_choice(grid)
    elif isinstance(grid, dict):
        return {
            key: sample_grid(value)
            for key, value in grid.iteritems()
        }
    else:
        raise ValueError('cannot sample from grid: {}'.format(grid))


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
    numpy.random.shuffle(featureid_to_kindid)
    return featureid_to_kindid


def generate_features(feature_count, feature_type='mixed'):
    if feature_type == 'mixed':
        feature_types = loom.schema.MODELS.keys()
    else:
        feature_types = [feature_type]
    features = []
    for feature_type in feature_types:
        module = loom.schema.MODELS[feature_type]
        for example in module.EXAMPLES:
            features.append(module.Shared.from_dict(example['shared']))
    features *= (feature_count + len(features) - 1) / len(features)
    numpy.random.shuffle(features)
    features = features[:feature_count]
    assert len(features) == feature_count
    loom.schema.sort_features(features)
    return features


def import_features(encoders):
    features = []
    for encoder in encoders:
        feature_type = encoder['model']
        feature = loom.schema.MODELS[feature_type].Shared()
        if feature_type in ['bb', 'gp', 'nich']:
            raw = sample_grid(loom.hyperprior.DEFAULTS[feature_type])
        elif feature_type == 'dpd':
            raw = sample_grid(loom.hyperprior.DEFAULTS[feature_type])
            raw['beta0'] = 1.0
            raw['betas'] = {}
            raw['counts'] = {}
        elif feature_type == 'dd':
            grid = loom.hyperprior.DEFAULTS[feature_type]['alpha']
            dim = len(encoder['symbols'])
            raw = {'alphas': [sample_grid(grid) for _ in xrange(dim)]}
        else:
            raise ValueError('unknown model: {}'.format(feature_type))
        feature.load(raw)
        features.append(feature)
    return features


def generate_model(features):
    featureid_to_kindid = generate_kinds(len(features))
    kind_count = 1 + max(featureid_to_kindid)
    cross_cat = loom.schema_pb2.CrossCat()
    kinds = [cross_cat.kinds.add() for _ in xrange(kind_count)]
    for kind in kinds:
        CLUSTERING.dump_protobuf(kind.product_model.clustering)
    for featureid, feature in enumerate(features):
        kindid = featureid_to_kindid[featureid]
        kind = kinds[kindid]
        feature_type = loom.schema.get_feature_type(feature)
        features = getattr(kind.product_model, feature_type)
        feature.dump_protobuf(features.add())
        kind.featureids.append(featureid)
    CLUSTERING.dump_protobuf(cross_cat.topology)
    loom.hyperprior.dump_default(cross_cat.hyper_prior)
    return cross_cat


@parsable.command
def generate(
        feature_type='mixed',
        row_count=1000,
        feature_count=100,
        density=0.5,
        rows_out='rows.pbs.gz',
        model_out='model.pb.gz',
        groups_out=None,
        init_out=None,
        debug=False,
        profile=None):
    '''
    Generate a synthetic dataset.
    '''
    root = os.getcwd()
    rows_out = os.path.abspath(rows_out)
    model_out = os.path.abspath(model_out)
    if init_out is not None:
        init_out = os.path.abspath(init_out)
    if groups_out is not None:
        groups_out = os.path.abspath(groups_out)

    features = generate_features(feature_count, feature_type)
    model = generate_model(features)

    with tempdir(cleanup_on_error=(not debug)):
        if init_out is None:
            init_out = os.path.abspath('init.pb.gz')
        with open_compressed(init_out, 'wb') as f:
            f.write(model.SerializeToString())

        config = {'generate': {'row_count': row_count, 'density': density}}
        config_in = os.path.abspath('config.pb.gz')
        loom.config.config_dump(config, config_in)

        os.chdir(root)
        loom.runner.generate(
            config_in=config_in,
            model_in=init_out,
            rows_out=rows_out,
            model_out=model_out,
            groups_out=groups_out,
            debug=debug,
            profile=profile)


@parsable.command
def generate_init(encoding_in, model_out, seed=0):
    '''
    Generate an initial model for inference.
    '''
    numpy.random.seed(seed)
    encoders = json_load(encoding_in)
    features = import_features(encoders)
    cross_cat = generate_model(features)
    with open_compressed(model_out, 'wb') as f:
        f.write(cross_cat.SerializeToString())


if __name__ == '__main__':
    parsable.dispatch()
