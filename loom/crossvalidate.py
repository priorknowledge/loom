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
import sys
from itertools import izip
import numpy
import numpy.random
from distributions.io.stream import (
    protobuf_stream_load,
    protobuf_stream_dump,
    json_dump,
)
from loom.util import mkdir_p
import loom.store
import loom.config
import loom.runner
import loom.generate
import loom.query
import parsable
parsable = parsable.Parsable()


def LOG(message):
    sys.stdout.write(message)
    sys.stdout.flush()


@parsable.command
def crossvalidate(
        name=None,
        sample_count=10,
        portion=0.9,
        extra_passes=loom.config.DEFAULTS['schedule']['extra_passes'],
        debug=False):
    '''
    Randomly split dataset; train models; score held-out data.
    '''
    assert 0 < portion and portion < 1, portion
    assert sample_count > 0, sample_count
    loom.store.require(name, 'encoding', 'tares', 'diffs', 'config')
    dataset = loom.store.get_paths(name, 'data')

    row_count = sum(1 for _ in protobuf_stream_load(dataset['diffs']))
    assert row_count > 1, 'too few rows to crossvalidate: {}'.format(row_count)
    train_count = max(1, min(row_count - 1, int(round(portion * row_count))))
    test_count = row_count - train_count
    assert 1 <= train_count and 1 <= test_count
    split = [True] * train_count + [False] * test_count

    mean_scores = []
    for seed in xrange(sample_count):
        LOG('running seed {}:'.format(seed))
        results = loom.store.get_paths(name, 'crossvalidate', seed)
        mkdir_p(results['root'])
        results['train'] = os.path.join(results['root'], 'train.pbs.gz')
        results['test'] = os.path.join(results['root'], 'test.pbs.gz')
        results['scores'] = os.path.join(results['root'], 'scores.json.gz')

        config = {
            'seed': seed,
            'schedule': {'extra_passes': extra_passes},
        }
        loom.config.config_dump(config, results['config'])

        numpy.random.seed(seed)
        numpy.random.shuffle(split)
        protobuf_stream_dump((
            row
            for s, row in izip(split, protobuf_stream_load(dataset['diffs']))
            if s
        ), results['train'])
        protobuf_stream_dump((
            row
            for s, row in izip(split, protobuf_stream_load(dataset['rows']))
            if not s
        ), results['test'])

        LOG(' shuffle')
        loom.runner.shuffle(
            rows_in=results['train'],
            rows_out=results['shuffled'],
            seed=seed,
            debug=debug)
        LOG(' init')
        loom.generate.generate_init(
            encoding_in=dataset['encoding'],
            model_out=results['init'],
            seed=seed)
        LOG(' infer')
        loom.runner.infer(
            config_in=results['config'],
            rows_in=results['shuffled'],
            tares_in=dataset['tares'],
            model_in=results['init'],
            model_out=results['model'],
            groups_out=results['groups'],
            debug=debug)
        LOG(' query')
        rows = loom.query.load_data_rows(results['test'])
        with loom.query.QueryServer(loom.query.SingleSampleProtobufServer(
                config_in=results['config'],
                model_in=results['model'],
                groups_in=results['groups'],
                debug=debug)) as query:
            scores = [query.score(row) for row in rows]

        json_dump(scores, results['scores'])
        mean_scores.append(numpy.mean(scores))
        LOG(' done\n')

    results = loom.store.get_paths(name, 'crossvalidate')
    results['scores'] = os.path.join(results['root'], 'scores.json.gz')
    json_dump(mean_scores, results['scores'])
    print 'score = {} +- {}'.format(
        numpy.mean(mean_scores),
        numpy.std(mean_scores))


if __name__ == '__main__':
    parsable.dispatch()
