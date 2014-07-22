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

import numpy
import loom.gridding

dd_alpha = numpy.logspace(-1, 2, 12).tolist()

pos_logspace = numpy.logspace(-8, 8, 100).tolist()
neg_logspace = (-numpy.logspace(-8, 8, 100)).tolist()

pitman_yor_grid = loom.gridding.pitman_yor()

DEFAULTS = {
    'topology': pitman_yor_grid,
    'clustering': pitman_yor_grid,
    'bb': {
        'alpha': dd_alpha,
        'beta': dd_alpha,
    },
    'dd': {
        'alpha': dd_alpha,
    },
    'dpd': {
        'gamma': (10 ** loom.gridding.left_heavy(-1, 2, 30)).tolist(),
        'alpha': (10 ** loom.gridding.right_heavy(-1, 1, 20)).tolist(),
    },
    'gp': {
        'alpha': numpy.logspace(-1, 5, 100).tolist(),
        'inv_beta': numpy.logspace(-5, 1, 100).tolist(),
    },
    'bnb': {
        'alpha': [2.0 ** p for p in xrange(-3, 1)],
        'beta': [2.0 ** p for p in xrange(13)],
        'r': [2 ** p for p in xrange(13)],
    },
    'nich': {
        'mu': neg_logspace + [0] + pos_logspace,
        'sigmasq': pos_logspace,
        'kappa': numpy.logspace(-2, 2, 30).tolist(),
        'nu': numpy.logspace(0, 2, 30).tolist(),
    },
}


def dump_default(message):
    loom.util.dict_to_protobuf(DEFAULTS, message)
