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
from nose.tools import assert_true, assert_equal
from loom.test.util import for_each_dataset, CLEANUP_ON_ERROR, assert_found
from distributions.fileutil import tempdir
from distributions.io.stream import open_compressed, protobuf_stream_load
from loom.schema_pb2 import ProductModel, CrossCat
import loom.config
import loom.runner

CONFIGS = [
    {
        'schedule': {'extra_passes': 0.0},
    },
    {
        'schedule': {'extra_passes': 1.5},
        'kernels': {
            'cat': {
                'empty_group_count': 1,
                'row_queue_capacity': 0,
            },
            'kind': {'iterations': 0},
        },
    },
    {
        'schedule': {'extra_passes': 1.5},
        'kernels': {
            'cat': {
                'empty_group_count': 1,
                'row_queue_capacity': 8,
            },
            'kind': {'iterations': 0},
        },
    },
    {
        'schedule': {'extra_passes': 1.5},
        'kernels': {
            'cat': {
                'empty_group_count': 1,
                'row_queue_capacity': 0,
            },
            'kind': {
                'iterations': 1,
                'empty_kind_count': 1,
                'row_queue_capacity': 0,
                'score_parallel': False,
            },
        },
    },
    {
        'schedule': {'extra_passes': 1.5, 'max_reject_iters': 1},
        'kernels': {
            'cat': {
                'empty_group_count': 1,
                'row_queue_capacity': 0,
            },
            'kind': {
                'iterations': 1,
                'empty_kind_count': 1,
                'row_queue_capacity': 0,
                'score_parallel': False,
            },
        },
    },
    {
        'schedule': {'extra_passes': 1.5, 'max_reject_iters': 1},
        'kernels': {
            'cat': {
                'empty_group_count': 1,
                'row_queue_capacity': 0,
            },
            'kind': {
                'iterations': 1,
                'empty_kind_count': 1,
                'row_queue_capacity': 0,
                'score_parallel': False,
            },
        },
    },
    {
        'schedule': {'extra_passes': 1.5, 'max_reject_iters': 100},
        'kernels': {
            'cat': {
                'empty_group_count': 1,
                'row_queue_capacity': 0,
            },
            'kind': {
                'iterations': 1,
                'empty_kind_count': 1,
                'row_queue_capacity': 0,
                'score_parallel': False,
            },
        },
    },
    {
        'schedule': {'extra_passes': 1.5, 'max_reject_iters': 100},
        'kernels': {
            'cat': {
                'empty_group_count': 1,
                'row_queue_capacity': 0,
            },
            'kind': {
                'iterations': 1,
                'empty_kind_count': 1,
                'row_queue_capacity': 8,
                'proposer_stage': False,
                'score_parallel': True,
            },
        },
    },
    {
        'schedule': {'extra_passes': 1.5, 'max_reject_iters': 100},
        'kernels': {
            'cat': {
                'empty_group_count': 1,
                'row_queue_capacity': 0,
            },
            'kind': {
                'iterations': 1,
                'empty_kind_count': 1,
                'row_queue_capacity': 8,
                'proposer_stage': True,
                'score_parallel': True,
            },
        },
    },
]


def get_group_counts(groups_out):
    group_counts = []
    for f in os.listdir(groups_out):
        group_count = 0
        groups = os.path.join(groups_out, f)
        for string in protobuf_stream_load(groups):
            group = ProductModel.Group()
            group.ParseFromString(string)
            group_count += 1
        group_counts.append(group_count)
    assert group_counts, 'no groups found'
    return group_counts


@for_each_dataset
def test_shuffle(rows, **unused):
    with tempdir(cleanup_on_error=CLEANUP_ON_ERROR):
        seed = 12345
        rows_out = os.path.abspath('rows_out.pbs.gz')
        loom.runner.shuffle(
            rows_in=rows,
            rows_out=rows_out,
            seed=seed)
        assert_found(rows_out)


@for_each_dataset
def test_infer(shuffled, init, name, **unused):
    with tempdir(cleanup_on_error=CLEANUP_ON_ERROR):
        row_count = sum(1 for _ in protobuf_stream_load(shuffled))
        with open_compressed(init) as f:
            message = CrossCat()
            message.ParseFromString(f.read())
        kind_count = len(message.kinds)

        for config in CONFIGS:
            loom.config.fill_in_defaults(config)
            schedule = config['schedule']
            print 'config: {}'.format(config)

            greedy = (schedule['extra_passes'] == 0)
            kind_iters = config['kernels']['kind']['iterations']
            kind_structure_is_fixed = greedy or kind_iters == 0

            with tempdir(cleanup_on_error=CLEANUP_ON_ERROR):
                config_in = os.path.abspath('config.pb.gz')
                model_out = os.path.abspath('model.pb.gz')
                groups_out = os.path.abspath('groups')
                assign_out = os.path.abspath('assign.pbs.gz')
                log_out = os.path.abspath('log.pbs.gz')
                os.mkdir(groups_out)
                loom.config.config_dump(config, config_in)
                loom.runner.infer(
                    config_in=config_in,
                    rows_in=shuffled,
                    model_in=init,
                    model_out=model_out,
                    groups_out=groups_out,
                    assign_out=assign_out,
                    log_out=log_out,
                    debug=True,)

                if kind_structure_is_fixed:
                    assert_equal(len(os.listdir(groups_out)), kind_count)

                group_counts = get_group_counts(groups_out)

                assign_count = sum(1 for _ in protobuf_stream_load(assign_out))
                assert_equal(assign_count, row_count)

            print 'row_count: {}'.format(row_count)
            print 'group_counts: {}'.format(' '.join(map(str, group_counts)))
            for group_count in group_counts:
                assert_true(
                    group_count <= row_count,
                    'groups are all singletons')


@for_each_dataset
def test_posterior_enum(rows, init, **unused):
    with tempdir(cleanup_on_error=CLEANUP_ON_ERROR):
        config_in = os.path.abspath('config.pb.gz')
        config = {
            'posterior_enum': {
                'sample_count': 7,
            },
            'kernels': {
                'kind': {
                    'row_queue_capacity': 0,
                    'score_parallel': False,
                },
            },
        }
        loom.config.config_dump(config, config_in)
        assert_found(config_in)

        samples_out = os.path.abspath('samples.pbs.gz')
        loom.runner.posterior_enum(
            config_in=config_in,
            model_in=init,
            rows_in=rows,
            samples_out=samples_out,
            debug=True)
        assert_found(samples_out)
        actual_count = sum(1 for _ in protobuf_stream_load(samples_out))
        assert_equal(actual_count, config['posterior_enum']['sample_count'])


@for_each_dataset
def test_generate(init, **unused):
    for row_count in [0, 1, 100]:
        for density in [0.0, 0.5, 1.0]:
            with tempdir(cleanup_on_error=CLEANUP_ON_ERROR):
                config_in = os.path.abspath('config.pb.gz')
                config = {
                    'generate': {
                        'row_count': row_count,
                        'density': density,
                    },
                }
                loom.config.config_dump(config, config_in)
                assert_found(config_in)

                rows_out = os.path.abspath('rows.pbs.gz')
                model_out = os.path.abspath('model.pb.gz')
                groups_out = os.path.abspath('groups')
                loom.runner.generate(
                    config_in=config_in,
                    model_in=init,
                    rows_out=rows_out,
                    model_out=model_out,
                    groups_out=groups_out,
                    debug=True)
                assert_found(rows_out, model_out, groups_out)

                group_counts = get_group_counts(groups_out)
                print 'group_counts: {}'.format(
                    ' '.join(map(str, group_counts)))
