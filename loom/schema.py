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

from distributions.lp.models import bb, dd, dpd, gp, nich

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
