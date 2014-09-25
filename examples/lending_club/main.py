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
import re
import sys
import csv
import urllib
import subprocess
import datetime
import dateutil.parser
from StringIO import StringIO
import parsable
from collections import Counter
from itertools import izip
from distributions.io.stream import open_compressed, json_dump, json_load
import loom.util
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
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
NAME = os.path.join(DATA, 'results')
DOWNLOADS = os.path.join(DATA, 'downloads')
SCHEMA_JSON = os.path.join(DATA, 'rows_csv')
ROWS_CSV = os.path.join(DATA, 'rows_csv')
SCHEMA_JSON = os.path.join(DATA, 'schema.json')
RELATED = os.path.join(DATA, 'related.csv')
MIN_ROW_LENGTH = 10
ROW_COUNTS = {
    'LoanStats3a.csv': 39787,
    'LoanStats3b.csv': 197787,
    'LoanStats3c.csv': 138735,
}
FEATURES = os.path.join(DATA, 'features.{}.json')
NOW = datetime.datetime.now()
FEATURE_FREQ = 0.01
SAMPLE_COUNT = 10

dot_counter = 0


def set_matplotlib_headless():
    import matplotlib
    matplotlib.use('Agg')


def savefig(name):
    from matplotlib import pyplot
    for ext in ['pdf', 'png']:
        filename = os.path.join(DATA, '{}.{}'.format(name, ext))
        print 'saving', filename
        pyplot.savefig(filename)


def print_dot(every=1):
    global dot_counter
    dot_counter += 1
    if dot_counter >= every:
        sys.stdout.write('.')
        sys.stdout.flush()
        dot_counter = 0


@parsable.command
def download():
    '''
    Download datset from website.
    '''
    loom.util.mkdir_p(DOWNLOADS)
    with loom.util.chdir(DOWNLOADS):
        for filename in FILES:
            if not os.path.exists(filename):
                url = URL + filename
                print 'fetching {}'.format(url)
                urllib.urlretrieve(url, filename)
            subprocess.check_call(['unzip', '-n', filename])


def load_rows(filename):
    row_count = ROW_COUNTS[filename]
    with open_compressed(os.path.join('data', 'downloads', filename)) as f:
        reader = csv.reader(f)
        header = []
        while len(header) < MIN_ROW_LENGTH:
            header = reader.next()
        for i, row in enumerate(reader):
            if i == row_count:
                raise StopIteration()
            if len(row) < MIN_ROW_LENGTH:
                continue
            yield header, row


@parsable.command
def explore_schema(count=1):
    '''
    Print header with some example values.
    '''
    for filename in ROW_COUNTS:
        print '-' * 80
        print filename
        for i, (header, row) in enumerate(load_rows(filename)):
            if i >= count:
                break
            for name, value in sorted(zip(header, row)):
                print name.rjust(20), value


def transform_string(string):
    return string.lower() if string else None


def transform_count(string):
    return int(string) if string else None


def transform_real(string):
    return float(string) if string else None


def transform_percent(string):
    return float(string.replace('%', '')) * 0.01 if string else None


def transform_sparse_real(string):
    if string:
        x = float(string)
        return [True, x] if x else [False, None]
    else:
        return [None, None]


def transform_maybe_count(string):
    return [True, int(string)] if string else [False, None]


def transform_days_ago(string):
    if string:
        delta = NOW - dateutil.parser.parse(string)
        return delta.total_seconds() / (24 * 60 * 60)
    else:
        return None


re_word = re.compile('[A-Za-z]{2,}')


def get_word_set(text):
    return frozenset(m.group().lower() for m in re_word.finditer(text))


def plot_text_features(field, counts, save=True):
    if save:
        set_matplotlib_headless()
    from matplotlib import pyplot
    counts = [count for word, count in counts]
    scale = 1.0 / max(counts)
    Y = [scale * count for count in counts]
    X = range(1, 1 + len(Y))
    pyplot.figure()
    pyplot.xscale('log')
    pyplot.yscale('log')
    pyplot.scatter(X, Y, alpha=0.3, edgecolors='none')
    pyplot.title('Distribution of {} words'.format(field))
    pyplot.xlabel('word rank')
    pyplot.ylabel('frequency relative to max')
    pyplot.grid(color='gray')
    pyplot.tight_layout()
    if save:
        savefig('text_fields.{}'.format(field))
    else:
        pyplot.show()


@parsable.command
def find_text_features(field, feature_freq=FEATURE_FREQ, plot=True):
    '''
    Build a list of most-common words to extract as sparse boolean features.
    '''
    print 'finding most common words in text field:', field
    counter = Counter()
    for filename in ROW_COUNTS:
        for header, row in load_rows(filename):
            pos = header.index(field)
            break
        for header, row in load_rows(filename):
            text = row[pos]
            counter.update(get_word_set(text))
            print_dot(1000)
    counts = counter.most_common()
    max_count = counts[0][1]
    min_count = feature_freq * max_count
    features = [word for word, count in counts if count > min_count]
    print 'restricting to {}/{} words for {}'.format(
        len(features),
        len(counts),
        field)
    json_dump(features, FEATURES.format(field))
    if plot:
        plot_text_features(field, counter.most_common())
    DATATYPES['text_{}'.format(field)] = text_datatype(field)


def transform_text(features):
    nones = [None] * len(features)
    zeros = [0] * len(features)
    index = {word: pos for pos, word in enumerate(features)}

    def transform(text):
        words = get_word_set(text)
        if words:
            result = zeros[:]
            for word in words:
                try:
                    result[index[word]] = 1
                except KeyError:
                    pass
            return [True] + result
        else:
            return [False] + nones

    return transform


def text_datatype(field):
    filename = FEATURES.format(field)
    if not os.path.exists(filename):
        print 'first run `python main.py find_text_features {}`'.format(field)
        return None
    else:
        features = json_load(filename)
        assert 'nonzero' not in features, 'name collision'
        return {
            'parts': ['nonzero'] + features,
            'model': ['bb'] + ['bb'] * len(features),
            'transform': transform_text(features),
        }


DATATYPES = {
    'count': {'model': 'gp', 'transform': transform_count},
    'real': {'model': 'nich', 'transform': transform_real},
    'date': {'model': 'nich', 'transform': transform_days_ago},
    'percent': {'model': 'nich', 'transform': transform_percent},
    'categorical': {'model': 'dd', 'transform': transform_string},
    'unbounded_categorical': {'model': 'dpd', 'transform': transform_string},
    'maybe_count': {
        'parts': ['nonzero', 'value'],
        'model': ['bb', 'gp'],
        'transform': transform_maybe_count,
    },
    'sparse_real': {
        'parts': ['nonzero', 'value'],
        'model': ['bb', 'nich'],
        'transform': transform_sparse_real,
    },
    'text_desc': text_datatype('desc'),
    'text_emp_title': text_datatype('emp_title'),
    'text_title': text_datatype('title'),
}


SCHEMA_IN = {
    'acc_now_delinq': 'count',
    'acc_open_past_24mths': 'count',
    'accept_d': 'date',
    'addr_city': 'unbounded_categorica',
    'addr_state': 'categorical',
    'annual_inc': 'real',
    'avg_cur_bal': 'real',
    'bc_open_to_buy': 'real',
    'bc_util': 'real',
    'chargeoff_within_12_mths': 'count',
    'collection_recovery_fee': 'sparse_real',
    'collections_12_mths_ex_med': 'count',
    'delinq_2yrs': 'count',
    'delinq_amnt': 'count',
    'desc': 'text_desc',
    'dti': 'real',
    'earliest_cr_line': 'date',
    'emp_length': 'categorical',
    'emp_title': 'text_emp_title',
    'exp_d': 'date',
    'funded_amnt': 'real',
    'funded_amnt_inv': 'real',
    'grade': 'categorical',
    'home_ownership': 'categorical',
    'id': None,
    'initial_list_status': 'categorical',
    'inq_last_6mths': 'count',
    'installment': 'real',
    'int_rate': 'percent',
    'is_inc_v': 'categorical',
    'issue_d': 'date',
    'last_credit_pull_d': 'date',
    'last_pymnt_amnt': 'real',
    'last_pymnt_d': 'date',
    'list_d': 'date',
    'loan_amnt': 'real',
    'loan_status': 'categorical',
    'member_id': None,
    'mo_sin_old_il_acct': 'count',
    'mo_sin_old_rev_tl_op': 'count',
    'mo_sin_rcnt_rev_tl_op': 'count',
    'mo_sin_rcnt_tl': 'count',
    'mort_acc': 'count',
    'mths_since_last_delinq': 'maybe_count',
    'mths_since_last_major_derog': 'maybe_count',
    'mths_since_last_record': 'maybe_count',
    'mths_since_recent_bc': 'maybe_count',
    'mths_since_recent_bc_dlq': 'maybe_count',
    'mths_since_recent_inq': 'maybe_count',
    'mths_since_recent_revol_delinq': 'maybe_count',
    'next_pymnt_d': 'date',
    'num_accts_ever_120_pd': 'count',
    'num_actv_bc_tl': 'count',
    'num_actv_rev_tl': 'count',
    'num_bc_sats': 'count',
    'num_bc_tl': 'count',
    'num_il_tl': 'count',
    'num_op_rev_tl': 'count',
    'num_rev_accts': 'count',
    'num_rev_tl_bal_gt_0': 'count',
    'num_sats': 'count',
    'num_tl_120dpd_2m': 'count',
    'num_tl_30dpd': 'count',
    'num_tl_90g_dpd_24m': 'count',
    'num_tl_op_past_12m': 'count',
    'open_acc': 'count',
    'out_prncp': 'real',
    'out_prncp_inv': 'real',
    'pct_tl_nvr_dlq': 'real',
    'percent_bc_gt_75': 'real',
    'policy_code': 'categorical',
    'pub_rec': 'maybe_count',
    'pub_rec_bankruptcies': 'maybe_count',
    'purpose': 'categorical',
    'pymnt_plan': 'categorical',
    'recoveries': 'sparse_real',
    'revol_bal': 'real',
    'revol_util': 'percent',
    'sub_grade': 'categorical',
    'tax_liens': 'count',
    'term': 'categorical',
    'title': 'text_title',
    'tot_coll_amt': 'sparse_real',
    'tot_cur_bal': 'sparse_real',
    'tot_hi_cred_lim': 'real',
    'total_acc': 'count',
    'total_bal_ex_mort': 'real',
    'total_bc_limit': 'real',
    'total_il_high_credit_limit': 'real',
    'total_pymnt': 'real',
    'total_pymnt_inv': 'real',
    'total_rec_int': 'real',
    'total_rec_late_fee': 'real',
    'total_rec_prncp': 'real',
    'total_rev_hi_lim': 'real',
    'url': None,
}


def transform_schema():
    schema = {}
    for key in SCHEMA_IN:
        datatype = DATATYPES.get(SCHEMA_IN[key])
        if datatype is not None:
            model = datatype['model']
            if 'parts' in datatype:
                for part, model_part in izip(datatype['parts'], model):
                    schema['{}_{}'.format(key, part)] = model_part
            else:
                schema[key] = model
    print 'transformed {} -> {} features'.format(len(SCHEMA_IN), len(schema))
    return schema


def transform_header():
    header = []
    for key in sorted(SCHEMA_IN.keys()):
        datatype = DATATYPES.get(SCHEMA_IN[key])
        if datatype is not None:
            if 'parts' in datatype:
                for part in datatype['parts']:
                    header.append('{}_{}'.format(key, part))
            else:
                header.append(key)
    return header


def transform_row(header_in, row_in):
    row_out = {}
    assert len(header_in) == len(row_in), row_in
    for key, value in izip(header_in, row_in):
        datatype = DATATYPES.get(SCHEMA_IN[key])
        if datatype is not None:
            try:
                value = datatype['transform'](value)
            except Exception as e:
                sys.stdout.write('{}: {}'.format(key, e))
                sys.stdout.write('\n')
            if 'parts' in datatype:
                for part, value_part in izip(datatype['parts'], value):
                    row_out['{}_{}'.format(key, part)] = value_part
            else:
                row_out[key] = value
    return row_out


def transform_rows(filename):
    header = transform_header()
    filename_out = os.path.join(ROWS_CSV, filename + '.gz')
    with open_compressed(filename_out, 'wb') as f:
        print 'writing', filename_out
        writer = csv.writer(f)
        writer.writerow(header)
        for header_in, row_in in load_rows(filename):
            row_out = transform_row(header_in, row_in)
            writer.writerow([row_out[key] for key in header])
            print_dot(1000)


@parsable.command
def transform():
    '''
    Transform dataset.
    '''
    schema = transform_schema()
    print 'writing', SCHEMA_JSON
    json_dump(schema, SCHEMA_JSON)
    loom.util.mkdir_p(ROWS_CSV)
    loom.util.parallel_map(transform_rows, ROW_COUNTS.keys())


@parsable.command
def ingest():
    '''
    Ingest dataset into loom.
    '''
    loom.tasks.ingest(NAME, SCHEMA_JSON, ROWS_CSV)


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
def plot_related(save=False):
    '''
    Plot results, either saving to file or displaying.
    '''
    if save:
        set_matplotlib_headless()
    import pandas
    import scipy.spatial
    import scipy.cluster
    from matplotlib import pyplot

    print 'loading data'
    df = pandas.read_csv(RELATED, index_col=0)
    matrix = df.as_matrix()

    print 'sorting features'
    dist = scipy.spatial.distance.pdist(matrix)
    clust = scipy.cluster.hierarchy.complete(dist)
    order = scipy.cluster.hierarchy.leaves_list(clust)
    sorted_matrix = matrix[order].T[order].T
    sorted_labels = [df.index[i].replace('_', ' ') for i in order]

    print 'plotting'
    pyplot.figure(figsize=(18, 18))
    pyplot.imshow(
        sorted_matrix,
        origin='lower',
        interpolation='none',
        cmap=pyplot.get_cmap('Greens'))
    dim = len(matrix)
    ticks = range(dim)
    fontsize = 1200.0 / dim
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
        vs='pub_rec_bankruptcies_nonzero',
        count=1000,
        save=False):
    '''
    Make some example predictions.
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
    pyplot.title('{} versus {}'.format(target, vs))
    pyplot.xlabel('probability')
    pyplot.tight_layout()

    if save:
        savefig('predict')
    else:
        pyplot.show()


@parsable.command
def run():
    '''
    Run entire pipeline:
        download
        find text features
        transform
        ingest
        infer
        related
        plot_related
    '''
    set_matplotlib_headless()
    download()
    for field in ['desc', 'emp_title', 'title']:
        find_text_features(field)
    transform()
    ingest()
    infer()
    related()
    plot_related(save=True)
    predict(save=True)


if __name__ == '__main__':
    parsable.dispatch()
