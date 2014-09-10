import numpy
import pymetis
from collections import defaultdict, namedtuple
from distributions.io.stream import open_compressed
from loom.schema_pb2 import CrossCat
from loom.cFormat import assignment_stream_load
from loom.util import LoomError, parallel_map
import loom.store

Row = namedtuple('Row', ['row_id', 'group_id', 'confidence'])


def group(root, feature_name, parallel=False):
    paths = loom.store.get_paths(root, sample_count=None)
    map_ = parallel_map if parallel else map
    groupings = map_(group_sample, [
        (sample, feature_name)
        for sample in paths['samples']
    ])
    return group_reduce(groupings)


def group_sample((sample, featureid)):
    model = CrossCat()
    with open_compressed(sample['model']) as f:
        model.ParseFromString(f.read())
    for kindid, kind in enumerate(model.kinds):
        if featureid in kind.featureids:
            break
    grouping = defaultdict(lambda: [])
    for assignment in assignment_stream_load(sample['assign']):
        groupid = assignment.groupids(kindid)
        grouping[groupid].append(assignment.rowid)
    return grouping.values()


def group_reduce(groupings):
    return find_consensus_grouping(groupings)


def find_consensus_grouping(groupings, debug=False):
    '''
    This implements Strehl et al's Meta-Clustering Algorithm [1].

    Inputs:
        groupings - a list of lists of lists of object ids, for example

            [
                [                   # sample 0
                    [0, 1, 2],      # sample 0, group 0
                    [3, 4],         # sample 0, group 1
                    [5]             # sample 0, group 2
                ],
                [                   # sample 1
                    [0, 1],         # sample 1, group 0
                    [2, 3, 4, 5]    # sample 1, group 1
                ]
            ]

    References:
    [1] Alexander Strehl, Joydeep Ghosh, Claire Cardie (2002)
        "Cluster Ensembles - A Knowledge Reuse Framework
        for Combining Multiple Partitions"
        Journal of Machine Learning Research
        http://jmlr.csail.mit.edu/papers/volume3/strehl02a/strehl02a.pdf
    '''
    if not groupings:
        raise LoomError('tried to find consensus among zero groupings')

    # ------------------------------------------------------------------------
    # Set up consensus grouping problem

    allgroups = sum(groupings, [])
    objects = set(sum(allgroups, []))
    objects = sorted(list(objects))
    index = {item: i for i, item in enumerate(objects)}

    vertices = [numpy.array(map(index.__getitem__, g), dtype=numpy.intp)
                for g in allgroups]

    contains = numpy.zeros((len(vertices), len(objects)), dtype=numpy.float32)
    for v, vertex in enumerate(vertices):
        contains[v, vertex] = 1  # i.e. for u in vertex: contains[v, u] = i

    # We use the binary Jaccard measure for similarity
    overlap = numpy.dot(contains, contains.T)
    diag = overlap.diagonal()
    denom = (diag.reshape(len(vertices), 1) +
             diag.reshape(1, len(vertices)) - overlap)
    similarity = overlap / denom

    # ------------------------------------------------------------------------
    # Format for metis

    if not (similarity.max() <= 1):
        raise LoomError('similarity.max() = {}'.format(similarity.max()))
    similarity *= 2**16  # metis segfaults if this is too large
    int_similarity = numpy.zeros(similarity.shape, dtype=numpy.int32)
    numpy.rint(similarity, out=int_similarity)

    edges = int_similarity.nonzero()
    edge_weights = map(int, int_similarity[edges])
    edges = numpy.transpose(edges)

    adjacency = [[] for _ in vertices]
    for i, j in edges:
        adjacency[i].append(j)

    # FIXME is there a better way to choose the final group count?
    group_count = int(numpy.median(map(len, groupings)))

    metis_args = {
        'nparts': group_count,
        'adjacency': adjacency,
        'eweights': edge_weights,
    }

    edge_cut, partition = pymetis.part_graph(**metis_args)

    # ------------------------------------------------------------------------
    # Clean up solution

    parts = range(group_count)
    if len(partition) != len(vertices):
        raise LoomError('metis output vector has wrong length')

    represents = numpy.zeros((len(parts), len(vertices)))
    for v, p in enumerate(partition):
        represents[p, v] = 1

    contains = numpy.dot(represents, contains)
    represent_counts = represents.sum(axis=1)
    represent_counts[numpy.where(represent_counts == 0)] = 1  # avoid NANs
    contains /= represent_counts.reshape(group_count, 1)

    bestmatch = contains.argmax(axis=0)
    confidence = contains[bestmatch, range(len(bestmatch))]
    if not all(numpy.isfinite(confidence)):
        raise LoomError('confidence is nan')

    nonempty_groups = sorted(list(set(bestmatch)))
    reindex = {j: i for i, j in enumerate(nonempty_groups)}

    grouping = [
        Row(row_id=objects[i], group_id=reindex[g], confidence=c)
        for i, (g, c) in enumerate(zip(bestmatch, confidence))
    ]

    group_ids = set(row.group_id for row in grouping)
    groups = [
        [row for row in grouping if row.group_id == gid]
        for gid in group_ids
    ]
    groups.sort(key=len, reverse=True)
    groups = [
        Row(row_id=row.row_id, group_id=group_id, confidence=row.confidence)
        for group_id, group in enumerate(groups)
        for row in group
    ]
    grouping.sort(key=lambda x: (x.group_id, -x.confidence, x.row_id))

    return grouping
