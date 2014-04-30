import os
import sys
import math
import numpy
from collections import defaultdict
from itertools import product
from nose import SkipTest
from nose.tools import assert_true, assert_equal, assert_greater
from distributions.tests.util import seed_all
from distributions.util import scores_to_probs, multinomial_goodness_of_fit
from distributions.fileutil import tempdir
from distributions.io.stream import json_load, json_dump, protobuf_stream_load
import loom.format
import loom.runner
import loom.util

try:
    import ccdb.generate
    import ccdb.sparse
    import ccdb.binary
    import ccdb.enumerate
    import ccdb.compare
except ImportError:
    raise SkipTest('FIXME ccdb needs distributions 1.0')

try:
    import tardis  # TODO remove dependency
except ImportError:
    raise SkipTest('tardis is not available')

SAMPLE_COUNT = 2000
TOPN = 35
CUTOFF = 1e-3
SEED = 123456789

# There is no clear reason to expect datatype to matter in posterior
# enumeration tests.  We run NICH because it is fast; anecdotally GP may be
# more sensitive in catching bugs.  Errors in other datatypes should be caught
# by other tests.
DATATYPES = [
    'NICH',
    #'GP',
]

# This list was suggested by suggest_small_datasets below.
# For more suggestions, run python test_posterior_enum.py
DIMENSIONS = [
    (6, 1),  # does not test kind kernel
    (5, 2),
    (3, 3),
    (2, 4),
    (1, 6),
    # LARGE
    (3, 4), (4, 3),
    # SHORT
    #(1, 3),
    #(1, 4),
    # NARROW
    #(1, 2),
    #(2, 2),
    #(3, 2),
    #(4, 2),
    # TINY
    (3, 2), (2, 3),
    # HUGE, under 300k cells
    #(2,8), (3,6), (4,4), (5,3), (6,2),
]

# Note: 1 = all sparse = no data.  0 = not sparse = dense = all data.
SPARSITIES = [
    1.0,
    0.5,
    0.0,
]

TYPEINFO = {
    'NICH': {'model': 'NormalInverseChiSq'},
    'ADD2': {'model': 'AsymmetricDirichletDiscrete', 'parameters': {'D': 2}},
    'ADD13': {'model': 'AsymmetricDirichletDiscrete', 'parameters': {'D': 13}},
    'GP': {'model': 'GP'},
    'DPM': {'model': 'DPM'},
}


def test():
    datasets = map(list, product(DIMENSIONS, DATATYPES, SPARSITIES))
    loom.util.parallel_map(_test_dataset, datasets)


def _test_dataset((dim, datatype, sparsity)):
    seed_all(SEED)
    object_count, feature_count = dim
    config = [object_count, feature_count, datatype, sparsity]
    suffix = '+'.join(map(str, config))
    with tempdir():
        basename = os.path.abspath(suffix)
        os.makedirs(basename)

        filenames = ['meta.json', 'data.bin', 'mask.bin', 'latent.json']
        meta_name, data_name, mask_name, latent_name = [
            os.path.join(basename, f) for f in filenames
        ]
        meta, data, mask, latent = generate_data(
            object_count,
            feature_count,
            datatype,
            sparsity)
        ccdb.binary.dump(meta, data, mask, meta_name, data_name, mask_name)
        json_dump(latent, latent_name)

        latent_scores = score_all_latents(
            meta_name,
            data_name,
            mask_name,
            latent_name)

        if feature_count == 1:
            kind_counts = [0]
        else:
            kind_counts = [0, 1, 2, 10]
        for kind_count in kind_counts:
            casename = '{}+{}'.format(kind_count, suffix)
            print 'Running {}'.format(casename)
            _test_dataset_config(
                casename,
                suffix,
                kind_count,
                meta_name,
                data_name,
                mask_name,
                latent_name,
                latent_scores)


def pretty_kind(kind):
    return '{} |{}|'.format(
        ' '.join(sorted(kind['features'])),
        '|'.join(sorted(' '.join(sorted(cat)) for cat in kind['categories'])))


def pretty_latent(latent):
    return ' - '.join(sorted(
        pretty_kind(kind)
        for kind in latent['structure']))


def _test_dataset_config(
        casename,
        kind_count,
        meta_name,
        data_name,
        mask_name,
        true_latent_name,
        latent_scores):

    meta = json_load(meta_name)
    samples = generate_samples(meta_name, data_name, mask_name, kind_count)
    probs = latent_scores['prob']

    # sorted, most prob to least prob
    sort_order = numpy.argsort(probs)[::-1]

    counts = defaultdict(lambda: 0)
    pretty_hashes = defaultdict(lambda: '?')
    for latent in samples:
        hash = ccdb.compare.latent_struct_hash(meta, latent)
        counts[hash] += 1
        if hash not in pretty_hashes:
            pretty_hashes[hash] = pretty_latent(latent)

    ALLN = len(sort_order)
    truncated = ALLN > TOPN
    true_probs = probs[sort_order][:TOPN]
    hashes_of_interest = latent_scores['hash'][sort_order][:TOPN]
    counts_list = [counts[h] for h in hashes_of_interest]

    goodness_of_fit = multinomial_goodness_of_fit(
        true_probs,
        counts_list,
        SAMPLE_COUNT,
        truncated=truncated)

    result = '{}, goodness of fit = {:0.3g}'.format(casename, goodness_of_fit)
    if goodness_of_fit > CUTOFF:
        print 'Passed {}'.format(result)
    else:
        print 'EXPECT\tACTUAL\tVALUE'
        for prob, count, hash in zip(
                true_probs,
                counts_list,
                hashes_of_interest):
            expect = prob * SAMPLE_COUNT
            pretty = pretty_hashes[hash]
            print '{:0.1f}\t{}\t{}'.format(expect, count, pretty)
        print 'Failed {}'.format(result)

    assert_greater(goodness_of_fit, CUTOFF, 'Failed {}'.format(result))


def generate_data(
        object_count,
        feature_count,
        datatype,
        sparsity,
        single_kind=False):
    typeinfo = TYPEINFO[datatype]
    meta = {
        'features': {'f%d' % d: typeinfo for d in xrange(feature_count)},
        'feature_pos': ['f%d' % d for d in xrange(feature_count)],
        'object_pos': ['o%d' % d for d in xrange(object_count)],
    }
    latent = {}
    if single_kind:
        latent['structure'] = [{'features': meta['feature_pos']}]
    data, mask, true_latent = ccdb.generate.generate(meta, latent)
    mask = ccdb.sparse.sparsify(mask, sparsity)
    return meta, data, mask, true_latent


def score_all_latents(
        meta_name,
        data_name,
        mask_name,
        true_latent_name,
        single_kind=False):
    meta = json_load(meta_name)
    true_latent = json_load(true_latent_name)

    many_kinds = not single_kind
    all_possible_latents = ccdb.enumerate.enumerate_latents(meta, many_kinds)
    dtype = [
        ('hash', '|S64'),
        ('score', numpy.float32),
        ('prob', numpy.float32),
    ]
    scores = numpy.zeros(len(all_possible_latents), dtype=dtype)
    for i, latent in enumerate(all_possible_latents):
        latent['hypers'] = true_latent['hypers']
        latent['model_hypers'] = true_latent['model_hypers']
        score = score_latent(latent, meta_name, data_name, mask_name)
        scores[i]['score'] = score
        scores[i]['hash'] = ccdb.compare.latent_struct_hash(meta, latent)
    scores['prob'][:] = scores_to_probs(scores['score'])
    return scores


def score_latent(latent, meta_name, data_name, mask_name):
    dp = tardis.CCDBDataProvider(meta_name, data_name, mask_name, True)
    tdb = tardis.TardisDB(dp)
    kf = tardis.KindFactory()
    latent_name = os.path.abspath('latent.json')
    with tempdir():
        json_dump(latent, latent_name)
        tardis.load_tardis_from_ccdb(
            meta_name,
            latent_name,
            dp.object_count(),
            tdb,
            kf,
            dp)
    pvs = [{'alpha': 1.0, 'd': 0.0}]
    ke = tardis.KindEvaluator(tdb, dp, kf, pvs)
    tardis_score = ke.total_score()
    assert_true(not math.isnan(tardis_score))
    return tardis_score


def generate_samples(model_name, rows_name, kind_count):
    with tempdir():
        samples_name = os.path.abspath('samples.pbs.gz')
        loom.runner.posterior_enum(
            model_name,
            rows_name,
            samples_name,
            SAMPLE_COUNT,
            kind_count)
        sample = loom.schema_pb2.PosteriorEnum.Sample()
        for string in protobuf_stream_load(samples_name):
            sample.ParseFromString(string)
            yield sample
            sample.Clear()


#-----------------------------------------------------------------------------
# dataset suggestions

def enum_partitions(count):
    if count == 0:
        yield []
    elif count == 1:
        yield [[1]]
    else:
        for p in enum_partitions(count - 1):
            yield p + [[count]]
            for i, part in enumerate(p):
                yield p[:i] + [part + [count]] + p[1 + i:]


BELL_NUMBERS = [1, 1, 2, 5, 15, 52, 203, 877, 4140, 21147, 115975]


def test_enum_partitions():
    for i, bell_number in enumerate(BELL_NUMBERS):
        count = 0
        for _ in enum_partitions(i):
            count += 1
        assert_equal(count, bell_number)


def count_crosscats(rows, cols):
    count = 0
    for kinds in enum_partitions(cols):
        count += BELL_NUMBERS[rows] ** len(kinds)
    return count


def suggest_small_datasets(max_count=300):
    enum_partitions
    max_rows = 10
    max_cols = 10
    print '=== Cross cat latent space sizes up to {} ==='.format(max_count)
    print '\t'.join('{} col'.format(cols) for cols in range(1, 1 + max_cols))
    print '-' * 8 * max_cols
    suggestions = {}
    for rows in range(1, 1 + max_rows):
        counts = []
        for cols in range(1, 1 + max_cols):
            count = count_crosscats(rows, cols)
            if count > max_count:
                suggestions[cols] = rows
                break
            counts.append(count)
        print '\t'.join(str(c) for c in counts)
    suggestions = ', '.join([
        '({},{})'.format(rows, cols)
        for cols, rows in suggestions.iteritems()
    ])
    print 'suggested test cases:', suggestions


if __name__ == '__main__':
    args = sys.argv[1:]
    if args:
        max_count = int(args[0])
    else:
        max_count = 300
    suggest_small_datasets(max_count)
