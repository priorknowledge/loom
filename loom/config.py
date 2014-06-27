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

from copy import deepcopy
import simplejson as json
from distributions.io.stream import open_compressed
import loom.schema_pb2

DEFAULTS = {
    'seed': 0,
    'target_mem_bytes': 4e9,
    'schedule': {
        'extra_passes': 500.0,
        'small_data_size': 4e3,
        'big_data_size': 1e9,
        'max_reject_iters': 100,
        'checkpoint_period_sec': 1e9,
    },
    'kernels': {
        'cat': {
            'empty_group_count': 1,
            'row_queue_capacity': 200,
            'parser_threads': 6,
        },
        'hyper': {
            'run': True,
            'parallel': True,
        },
        'kind': {
            'iterations': 32,
            'empty_kind_count': 32,
            'row_queue_capacity': 200,
            'parser_threads': 6,
            'score_parallel': True,
        },
    },
    'posterior_enum': {
        'sample_count': 100,
        'sample_skip': 10,
    },
    'generate': {
        'row_count': 100,
        'density': 0.5,
    },
}


def fill_in_defaults(config, defaults=DEFAULTS):
    assert isinstance(config, dict), config
    assert isinstance(defaults, dict), defaults
    for key, default in defaults.iteritems():
        if key not in config:
            config[key] = deepcopy(default)
        elif isinstance(default, dict):
            fill_in_defaults(config[key], default)


def protobuf_dump(config, message, warn='WARN ignoring config'):
    for key, value in config.iteritems():
        warn_key = '{}.{}'.format(warn, key) if warn else None
        if hasattr(message, key):
            if isinstance(value, dict):
                protobuf_dump(value, getattr(message, key), warn_key)
            else:
                setattr(message, key, value)
        elif warn:
            print warn_key


def config_dump(config, filename):
    config = deepcopy(config)
    fill_in_defaults(config)
    message = loom.schema_pb2.Config()
    protobuf_dump(config, message)
    with open_compressed(filename, 'wb') as f:
        f.write(message.SerializeToString())


if __name__ == '__main__':
    print json.dumps(DEFAULTS, indent=4, sort_keys=True)
