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
import random
import parsable
from distributions.lp.clustering import PitmanYor
from loom.util import tempdir
from distributions.io.stream import open_compressed
import loom.schema
import loom.hyperprior
import loom.config
import loom.runner
parsable = parsable.Parsable()

CLUSTERING = PitmanYor.from_dict({'alpha': 2.0, 'd': 0.1})


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


def generate_features(feature_count, feature_type='mixed'):
    if feature_type == 'mixed':
        feature_types = loom.schema.FEATURE_TYPES.keys()
    else:
        feature_types = [feature_type]
    features = []
    for feature_type in feature_types:
        module = loom.schema.FEATURE_TYPES[feature_type]
        for example in module.EXAMPLES:
            features.append(module.Shared.from_dict(example['shared']))
    features *= (feature_count + len(features) - 1) / len(features)
    random.shuffle(features)
    features = features[:feature_count]
    assert len(features) == feature_count
    loom.schema.sort_features(features)
    return features


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
        init_out='init.pb.gz',
        rows_out='rows.pbs.gz',
        model_out='model.pb.gz',
        groups_out='groups',
        debug=False,
        profile=None):
    '''
    Generate a synthetic dataset.
    '''
    root = os.path.abspath(os.path.curdir)
    init_out = os.path.abspath(init_out)
    rows_out = os.path.abspath(rows_out)
    model_out = os.path.abspath(model_out)
    groups_out = os.path.abspath(groups_out)

    model = generate_model(row_count, feature_count, feature_type, density)
    with open_compressed(init_out, 'w') as f:
        f.write(model.SerializeToString())

    with tempdir(cleanup_on_error=(not debug)):
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


if __name__ == '__main__':
    parsable.dispatch()
