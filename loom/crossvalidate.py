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
from itertools import izip
import numpy
import numpy.random
from distributions.io.stream import (
    protobuf_stream_load,
    protobuf_stream_dump,
    json_dump,
)
from loom.util import LOG
import loom.store
import loom.config
import loom.runner
import loom.generate
import loom.query
import parsable
parsable = parsable.Parsable()


def crossvalidate_one(
        seed,
        test_count,
        train_count,
        inputs,
        results,
        extra_passes,
        debug):
    LOG('running seed {}:'.format(seed))
    results['train'] = os.path.join(
        results['root'],
        'train',
        'diffs.pbs.gz')
    results['test'] = os.path.join(results['root'], 'test', 'rows.pbs.gz')
    results['scores'] = os.path.join(results['root'], 'scores.json.gz')

    config = {
        'seed': seed,
        'schedule': {'extra_passes': extra_passes},
    }
    loom.config.config_dump(config, results['samples'][0]['config'])

    numpy.random.seed(seed)
    split = [True] * train_count + [False] * test_count
    numpy.random.shuffle(split)
    diffs_in = protobuf_stream_load(inputs['ingest']['diffs'])
    protobuf_stream_dump(
        (row for s, row in izip(split, diffs_in) if s),
        results['train'])
    rows_in = protobuf_stream_load(inputs['ingest']['rows'])
    protobuf_stream_dump(
        (row for s, row in izip(split, rows_in) if not s),
        results['test'])

    LOG(' shuffle')
    loom.runner.shuffle(
        rows_in=results['train'],
        rows_out=results['samples'][0]['shuffled'],
        seed=seed,
        debug=debug)
    LOG(' init')
    loom.generate.generate_init(
        encoding_in=inputs['ingest']['encoding'],
        model_out=results['samples'][0]['init'],
        seed=seed)
    LOG(' infer')
    loom.runner.infer(
        config_in=results['samples'][0]['config'],
        rows_in=results['samples'][0]['shuffled'],
        tares_in=inputs['ingest']['tares'],
        model_in=results['samples'][0]['init'],
        model_out=results['samples'][0]['model'],
        groups_out=results['samples'][0]['groups'],
        debug=debug)
    LOG(' query')
    rows = loom.query.load_data_rows(results['test'])
    loom.config.query_config_dump({}, results['query']['config'])
    with loom.query.get_server(results['root'], debug=debug) as query:
        scores = [query.score(row) for row in rows]

    json_dump(scores, results['scores'])
    LOG(' done\n')
    return numpy.mean(scores)


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
    loom.store.require(name, [
        'ingest.encoding',
        'ingest.tares',
        'ingest.diffs',
    ])
    inputs = loom.store.get_paths(name)

    row_count = sum(1 for _ in protobuf_stream_load(inputs['ingest']['diffs']))
    assert row_count > 1, 'too few rows to crossvalidate: {}'.format(row_count)
    train_count = max(1, min(row_count - 1, int(round(portion * row_count))))
    test_count = row_count - train_count
    assert 1 <= train_count and 1 <= test_count

    mean_scores = []
    for seed in xrange(sample_count):
        results = loom.store.get_paths(
            os.path.join(name, 'crossvalidate/{}'.format(seed)))
        mean = crossvalidate_one(
            seed,
            test_count,
            train_count,
            inputs,
            results,
            extra_passes,
            debug)
        mean_scores.append(mean)

    results = loom.store.get_paths(os.path.join(name, 'crossvalidate'))
    results['scores'] = os.path.join(results['root'], 'scores.json.gz')
    json_dump(mean_scores, results['scores'])
    print 'score = {} +- {}'.format(
        numpy.mean(mean_scores),
        numpy.std(mean_scores))


if __name__ == '__main__':
    parsable.dispatch()
