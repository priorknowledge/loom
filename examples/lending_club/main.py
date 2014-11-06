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
import urllib
import subprocess
from contextlib2 import ExitStack
from StringIO import StringIO
import parsable
import loom.util
import loom.cleanse
import loom.tasks

# see https://www.lendingclub.com/info/download-data.action
URL = 'https://resources.lendingclub.com/'
FILES = [
    'LoanStats3a.csv.zip',
    'LoanStats3b.csv.zip',
    'LoanStats3c.csv.zip',
    # 'RejectStatsA.csv.zip',
    # 'RejectStatsB.csv.zip',
]
ROOT = os.path.dirname(os.path.abspath(__file__))
SCHEMA = os.path.join(ROOT, 'schema.csv')
DATA = os.path.join(ROOT, 'data')
NAME = os.path.join(DATA, 'results')
DOWNLOADS = os.path.join(DATA, 'downloads')
CLEANSED = os.path.join(DATA, 'cleansed')
RELATED = os.path.join(DATA, 'related.csv')
GROUP = os.path.join(DATA, 'group.{}.csv.gz')
MIN_ROW_LENGTH = 10
ROW_COUNTS = {
    'LoanStats3a.csv': 39787,
    'LoanStats3b.csv': 197787,
    'LoanStats3c.csv': 138735,
}
FEATURES = os.path.join(DATA, 'features.{}.json')
FEATURE_FREQ = 0.01
SAMPLE_COUNT = 10


def set_matplotlib_headless():
    import matplotlib
    matplotlib.use('Agg')


def savefig(name):
    from matplotlib import pyplot
    for ext in ['pdf', 'png', 'eps']:
        filename = os.path.join(DATA, '{}.{}'.format(name, ext))
        print 'saving', filename
        pyplot.savefig(filename)


@parsable.command
def download():
    '''
    Download datset from website; unzip.
    '''
    loom.util.mkdir_p(DOWNLOADS)
    with loom.util.chdir(DOWNLOADS):
        for filename in FILES:
            if not os.path.exists(filename):
                url = URL + filename
                print 'fetching {}'.format(url)
                urllib.urlretrieve(url, filename)
            subprocess.check_call(['unzip', '-n', filename])


def cleanse_one(filename):
    row_count = ROW_COUNTS[filename]
    source = os.path.join(DOWNLOADS, filename)
    destin = os.path.join(CLEANSED, filename + '.gz')
    print 'forcing ascii', filename
    loom.cleanse.force_ascii(source, destin)
    print 'truncating', filename
    with ExitStack() as stack:
        with_ = stack.enter_context
        temp = with_(loom.util.temp_copy(destin))
        writer = with_(loom.util.csv_writer(temp))
        reader = with_(loom.util.csv_reader(destin))
        count = 0
        for row in reader:
            if len(row) >= MIN_ROW_LENGTH:
                if count <= row_count:
                    writer.writerow(row)
                    count += 1
                else:
                    break


@parsable.command
def cleanse():
    '''
    Cleanse input files.
    '''
    loom.util.parallel_map(cleanse_one, ROW_COUNTS.keys())
    print 'repartitioning'
    loom.cleanse.repartition_csv_dir(CLEANSED)


def load_rows():
    rows_csv = loom.store.get_paths()['ingest']['rows_csv']
    filenames = os.listdir(rows_csv)
    for filename in filenames:
        with loom.util.csv_reader(os.path.join(rows_csv, filename)) as reader:
            header = reader.next()
            for row in reader:
                yield header, row


@parsable.command
def transform():
    '''
    Transform dataset.
    '''
    loom.tasks.transform(NAME, SCHEMA, CLEANSED)


@parsable.command
def ingest():
    '''
    Ingest dataset into loom.
    '''
    loom.tasks.ingest(NAME, id_field='id')


@parsable.command
def infer(sample_count=SAMPLE_COUNT):
    '''
    Infer model.
    '''
    loom.tasks.infer(NAME, sample_count=sample_count)


@parsable.command
def related():
    '''
    Compute pairwise feature relatedness.
    '''
    with loom.tasks.query(NAME) as preql:
        preql.relate(preql.feature_names, result_out=RELATED)


@parsable.command
def plot_related(target='loan_status', feature_count=100, save=False):
    '''
    Plot results, either saving to file or displaying.

    If target is None, plot all features;
    otherwise plot feature_count features most related to target;
    '''
    if save:
        set_matplotlib_headless()
    import pandas
    import scipy.spatial
    import scipy.cluster
    from matplotlib import pyplot

    print 'loading data'
    df = pandas.read_csv(RELATED, index_col=0)
    if target and target.lower() != 'none':
        df = df.sort(target, ascending=False)[:feature_count].transpose()
        df = df.sort(target, ascending=False)[:feature_count].copy()

    print 'sorting features'
    matrix = df.as_matrix()
    dist = scipy.spatial.distance.pdist(matrix)
    clust = scipy.cluster.hierarchy.complete(dist)
    order = scipy.cluster.hierarchy.leaves_list(clust)
    sorted_matrix = matrix[order].T[order].T
    sorted_labels = [df.index[i].replace('_', ' ') for i in order]

    print 'plotting'
    pyplot.figure(figsize=(18, 18))
    pyplot.imshow(
        sorted_matrix ** 0.5,
        origin='lower',
        interpolation='none',
        cmap=pyplot.get_cmap('Greens'))
    dim = len(matrix)
    ticks = range(dim)
    fontsize = 1200.0 / (dim + 20)
    pyplot.xticks(ticks, sorted_labels, fontsize=fontsize, rotation=90)
    pyplot.yticks(ticks, sorted_labels, fontsize=fontsize)
    pyplot.title('Pairwise Relatedness of {} Features'.format(dim))
    pyplot.tight_layout()

    if save:
        savefig('related')
    else:
        pyplot.show()


@parsable.command
def find_related(target='loan_status', count=30):
    '''
    Find features related to target feature.
    '''
    import pandas
    df = pandas.read_csv(RELATED, index_col=0)
    df.sort(target, ascending=False, inplace=True)
    print 'Top {} features related to {}:'.format(count, target)
    print df[:count][target]


@parsable.command
def predict(
        target='loan_status',
        vs='mths_since_recent_bc.nonzero',
        count=1000,
        save=False):
    '''
    Make some example predictions. Interesting features include:
        mths_since_recent_bc.nonzero
        emp_title.nonzero
        pub_rec_bankruptcies.nonzero
    '''
    if save:
        set_matplotlib_headless()
    import pandas
    from matplotlib import pyplot
    with loom.tasks.query(NAME) as preql:
        query = pandas.DataFrame({
            vs: [None, 'True', 'False'],
            target: [None, None, None],
        })
        result = StringIO(preql.predict(StringIO(query.to_csv()), count))

    df = pandas.read_csv(result)
    batch = df.columns[0]
    groups = df.groupby([batch, target]).size() / count
    counts = pandas.concat([groups[i] for i in range(3)], axis=1)
    counts.columns = ['unknown', 'present', 'absent']
    counts.sort('unknown', inplace=True)
    counts.plot(kind='barh')
    pyplot.grid()
    pyplot.title('{} -vs- {}'.format(target, vs))
    pyplot.xlabel('probability')
    pyplot.tight_layout()

    if save:
        savefig('predict')
    else:
        pyplot.show()


@parsable.command
def group(target='loan_status'):
    '''
    Compute row grouping by target column.
    '''
    with loom.tasks.query(NAME) as preql:
        preql.group(target, result_out=GROUP.format(target))


def select_rows(ids):
    ids = frozenset(ids)
    header, _ = load_rows().next()
    id_pos = header.index('id')
    return {row[id_pos]: row for _, row in load_rows() if row[id_pos] in ids}


@parsable.command
def print_groups(target='loan_status'):
    '''
    Print group sizes and typical row from each group.
    '''
    stats_list = []
    stats = None
    with loom.util.csv_reader(GROUP.format(target)) as reader:
        header = reader.next()
        assert header
        for row_id, group_id, confidence in reader:
            group_id = int(group_id)
            confidence = float(confidence)
            if stats and stats['group_id'] == group_id:
                stats['count'] += 1
                stats['sum'] += confidence
            else:
                stats = {
                    'group_id': group_id,
                    'example': row_id,
                    'count': 1,
                    'sum': confidence,
                }
                stats_list.append(stats)
    rows = select_rows(stats['example'] for stats in stats_list)
    for stats in stats_list:
        # FIXME the row should be csv-escaped
        stats['row'] = ','.join(rows[stats['example']])
    stats_list.sort(key=lambda stats: stats['sum'], reverse=True)
    # FIXME print in csv format, including header
    print ('{:>12}' * 4).format('count', 'weight', 'id', 'example')
    print '-' * 12 * 4
    for stats in stats_list:
        print '{count:>12}{sum:>12.1f}{group_id:>12}{example:>12}: '\
            '{row}'.format(**stats)


@parsable.command
def run(sample_count=SAMPLE_COUNT):
    '''
    Run entire pipeline:
        download
        cleanse
        transform
        ingest
        infer
        related
        plot_related
        predict
    '''
    set_matplotlib_headless()
    download()
    cleanse()
    transform()
    ingest()
    infer(sample_count=sample_count)
    related()
    plot_related(save=True)
    predict(save=True)
    group()


if __name__ == '__main__':
    parsable.dispatch()
