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
import copy
from itertools import izip
import pymetis
from distributions.io.stream import json_load
import distributions.lp.clustering
import loom.group
from loom.group import METIS_ARGS_TEMPFILE
from loom.group import find_consensus_grouping
from nose.tools import assert_almost_equal
from nose.tools import assert_equal
from nose.tools import assert_set_equal


def test_metis():

    if os.path.exists(METIS_ARGS_TEMPFILE):
        print 'Loading metis args from %s' % METIS_ARGS_TEMPFILE
        args = json_load(METIS_ARGS_TEMPFILE)

    else:
        print 'Using simple metis args'
        args = {
            'nparts': 2,
            'adjacency': [[0, 2, 3], [1, 2], [0, 1, 2], [0, 3]],
            'eweights': [1073741824, 429496736, 357913952, 1073741824,
                         536870912, 429496736, 536870912, 1073741824,
                         357913952, 1073741824],
        }

    assert len(args['eweights']) == sum(map(len, args['adjacency']))

    print 'Running unweighted metis...'
    unweighted = dict(args)
    del unweighted['eweights']
    edge_cut, partition = pymetis.part_graph(**unweighted)
    print 'Finished unweighted metis'

    print 'Running metis...'
    edge_cut, partition = pymetis.part_graph(**args)
    print 'Finished metis'


class TestTypeIsCorrect:
    def __init__(self):

        ROW_COUNT = 1000
        SAMPLE_COUNT = 10

        self.clustering = distributions.lp.clustering.PitmanYor()
        self.sample_count = SAMPLE_COUNT
        self.row_ids = map(str, range(ROW_COUNT))

    def sample_grouping(self):
        assignments = self.clustering.sample_assignments(len(self.row_ids))
        return loom.group.collate(izip(assignments, self.row_ids))

    def sample_groupings(self):
        return [self.sample_grouping() for _ in xrange(self.sample_count)]

    def test_simple(self):
        groupings = self.sample_groupings()
        grouping = find_consensus_grouping(groupings, debug=True)
        assert isinstance(grouping, list)
        for row in grouping:
            assert isinstance(row, loom.group.Row), row

        row_ids = set(row.row_id for row in grouping)
        assert len(row_ids) == len(grouping), 'grouping had duplicate rows'
        assert_set_equal(set(self.row_ids), row_ids)

        group_ids = sorted(list(set(row.group_id for row in grouping)))
        assert_equal(
            group_ids,
            range(len(group_ids)),
            'group ids were not a contiguous range of integers')

    def test_sorting(self):
        for i in xrange(10):
            groupings = self.sample_groupings()
            grouping = find_consensus_grouping(groupings, debug=True)
            assert_equal(
                grouping,
                sorted(
                    grouping,
                    key=lambda x: (x.group_id, -x.confidence, x.row_id)))

            group_ids = sorted(set(row.group_id for row in grouping))
            counts = [
                sum(1 for row in grouping if row.group_id == gid)
                for gid in group_ids
            ]
            assert_equal(counts, sorted(counts, reverse=True))


class TestValueIsCorrect:
    def __init__(self):

        LEVELS = 5
        # LEVELS = 6  # XXX FIXME 6 or more levels fails

        self.row_ids = []
        self._grouping = []
        for i in range(0, LEVELS):
            level = range(2 ** i - 1, 2 ** (i + 1) - 1)
            # level = sorted(map(str, level))
            self.row_ids += level
            self._grouping.append(level)

    @property
    def grouping(self):
        return copy.deepcopy(self._grouping)

    def _assert_correct(self, grouping, confidence=None):

        if confidence is not None:
            for row in grouping:
                assert_almost_equal(row.confidence, confidence)

        grouping.sort(key=(lambda r: r.row_id))

        groups = loom.group.collate(
            (row.group_id, row.row_id)
            for row in grouping
        )
        groups.sort(key=len)
        for group in groups:
            group.sort()

        assert_equal(groups, self.grouping)

    def test_correct_on_perfect_data(self):
        for sample_count in range(1, 11):
            groupings = [self.grouping] * sample_count
            grouping = find_consensus_grouping(groupings)
            self._assert_correct(grouping, confidence=1.0)

    def test_correct_on_noisy_data(self):
        SAMPLE_COUNT = 10
        GROUP_COUNT = len(self.grouping)

        object_index = {
            o: g
            for g, group in enumerate(self.grouping)
            for o in group
        }

        # each object is in the wrong place in one grouping
        groupings = []
        for g in range(SAMPLE_COUNT):
            groups = self.grouping
            for o in self.row_ids[g::SAMPLE_COUNT]:
                t = object_index[o]
                f = (t + 1) % GROUP_COUNT
                groups[t].remove(o)
                groups[f].append(o)
            groups = filter(len, groups)
            groupings.append(groups)

        grouping = find_consensus_grouping(groupings)
        self._assert_correct(grouping)

    def test_correct_despite_outliers(self):
        SAMPLE_COUNT = 10

        fine = [[o] for o in self.row_ids]
        coarse = [[o for o in self.row_ids]]
        groupings = [fine, coarse] + [self.grouping] * (SAMPLE_COUNT - 2)

        grouping = find_consensus_grouping(groupings)
        self._assert_correct(grouping)


if __name__ == '__main__':
    test_metis()
