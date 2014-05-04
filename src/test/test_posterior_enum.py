import os
import sys
import numpy
from collections import defaultdict
from itertools import izip, product
from nose import SkipTest
from nose.tools import assert_equal
import numpy.random
from distributions.tests.util import seed_all
from distributions.util import scores_to_probs
from distributions.fileutil import tempdir
from distributions.io.stream import protobuf_stream_load, protobuf_stream_dump
from distributions.lp.models import dd, dpd, nich, gp
from distributions.lp.clustering import PitmanYor
from distributions.tests.util import assert_counts_match_probs
import loom.schema_pb2
import loom.runner
import loom.util

assert dd and dpd and gp and nich  # pacify pyflakes

CLEANUP_ON_ERROR = int(os.environ.get('CLEANUP_ON_ERROR', 1))

SAMPLE_COUNT = 10000
SEED = 123456789

CLUSTERING = PitmanYor.from_dict({'alpha': 1.0, 'd': 0.5})

# There is no clear reason to expect feature_type to matter in posterior
# enumeration tests.  We run NICH because it is fast; anecdotally GP may be
# more sensitive in catching bugs.  Errors in other feature_types should be
# caught by other tests.
FEATURE_TYPES = {
    #'dd': dd,
    #'dpd': dpd,
    'nich': nich,
    #'gp': gp,
}

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

DENSITIES = [
    1.0,
    0.5,
    0.0,
]


def test():
    datasets = map(list, product(DIMENSIONS, FEATURE_TYPES, DENSITIES))
    loom.util.parallel_map(_test_dataset, datasets)


def _test_dataset((dim, feature_type, density)):
    seed_all(SEED)
    object_count, feature_count = dim
    with tempdir(cleanup_on_error=CLEANUP_ON_ERROR):

        model_name = os.path.abspath('model.pb')
        rows_name = os.path.abspath('rows.pbs')

        model = generate_model(feature_count, feature_type)
        dump_model(model, model_name)

        rows = generate_rows(
            object_count,
            feature_count,
            feature_type,
            density)
        dump_rows(rows, rows_name)

        if feature_count == 1:
            configs = [{'kind_count': 0, 'kind_iters': 0}]
        else:
            configs = [
                {'kind_count': 1, 'kind_iters': 10},
                {'kind_count': 10, 'kind_iters': 1},
            ]

        for config in configs:
            print 'Running {} {} {} {} {}'.format(
                object_count,
                feature_count,
                feature_type,
                density,
                config)
            _test_dataset_config(rows_name, model_name, config)


def _test_dataset_config(model_name, rows_name, config):
    counts = defaultdict(lambda: 0)
    scores = {}
    for sample, score in generate_samples(model_name, rows_name, config):
        counts[sample] += 1
        scores[sample] = score
    keys = scores.keys()
    scores_list = [scores[key] for key in keys]
    probs_list = scores_to_probs(scores_list)
    probs = {key: prob for key, prob in izip(keys, probs_list)}
    assert_counts_match_probs(counts, probs)


def generate_model(feature_count, feature_type):
    #module = FEATURE_TYPES[feature_type]
    #shared = module.Shared.from_dict(module.EXAMPLES[0])
    #message = loom.schema_pb2.CrossCat()
    raise SkipTest('TODO')


def dump_model(model, model_name):
    with open(model_name, 'wb') as f:
        f.write(model.SerializeToString())


def generate_rows(object_count, feature_count, feature_type, density):
    assert object_count >= 0
    assert feature_count >= 0
    assert 0 <= density and density <= 1, density

    # generate structure
    feature_assignments = CLUSTERING.sample_assignments(feature_count)
    kind_count = len(set(feature_assignments))
    object_assignments = [
        CLUSTERING.sample_assignments(object_count)
        for _ in xrange(kind_count)
    ]
    group_counts = [
        len(set(assignments))
        for assignments in object_assignments
    ]

    # generate data
    module = FEATURE_TYPES[feature_type]
    shared = module.Shared.from_dict(module.EXAMPLES[0])

    def sampler_create():
        sampler = module.Sampler()
        sampler.init(shared)
        return sampler

    values = [[None] * feature_count for _ in xrange(object_count)]
    if density > 0:
        for f, k in enumerate(feature_assignments):
            samplers = [sampler_create() for _ in xrange(group_counts[k])]
            for i, g in enumerate(object_assignments[k]):
                if numpy.random.uniform() < density:
                    values[i][f] = samplers[g].eval()
    return values


def dump_rows(values, rows_name):
    row = loom.schema_pb2.SparseRow()

    def rows():
        for values_row in values:
            for value in values_row:
                if value is None:
                    row.add_observed(False)
                else:
                    row.add_observed(True)
                    if isinstance(value, bool):
                        row.add_booleans(value)
                    elif isinstance(value, int):
                        row.add_counts(value)
                    elif isinstance(value, float):
                        row.add_reals(value)
            yield row.SerializeToString()
            row.Clear()

    protobuf_stream_dump(rows, rows_name)


def parse_sample(message):
    feature_assignments = tuple(message.featureid_to_kindid)
    object_assignments = tuple(tuple(kind.groupids) for kind in message.kinds)
    return (feature_assignments, object_assignments)


def generate_samples(model_name, rows_name, config):
    with tempdir(cleanup_on_error=CLEANUP_ON_ERROR):
        samples_name = os.path.abspath('samples.pbs.gz')
        loom.runner.posterior_enum(
            model_name,
            rows_name,
            samples_name,
            SAMPLE_COUNT,
            **config)
        message = loom.schema_pb2.PosteriorEnum.Sample()
        count = 0
        for string in protobuf_stream_load(samples_name):
            message.ParseFromString(string)
            sample = parse_sample(message)
            score = float(message.score)
            yield sample, score
            message.Clear()
            count += 1
        assert count == SAMPLE_COUNT


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
        count = sum(1 for _ in enum_partitions(i))
        assert_equal(count, bell_number)


def count_crosscats(rows, cols):
    return sum(
        BELL_NUMBERS[rows] ** len(kinds)
        for kinds in enum_partitions(cols))


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
