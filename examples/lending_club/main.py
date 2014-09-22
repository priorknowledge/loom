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
import csv
import urllib
import subprocess
import datetime
import dateutil.parser
import parsable
from itertools import izip
from distributions.io.stream import open_compressed, json_dump
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
DATA = os.path.join(os.path.dirname(__file__), 'data')
DOWNLOADS = os.path.join(DATA, 'downloads')
SCHEMA_JSON = os.path.join(DATA, 'rows_csv')
ROWS_CSV = os.path.join(DATA, 'rows_csv')
SCHEMA_JSON = os.path.join(DATA, 'schema.json')
MIN_ROW_LENGTH = 10
ROW_COUNTS = {
    'LoanStats3a.csv': 39787,
    'LoanStats3b.csv': 197787,
    'LoanStats3c.csv': 138735,
}
NOW = datetime.datetime.now()



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
            subprocess.check_call(['unzip', filename])
            subprocess.check_call(['sed', '-i', '1d', filename])


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
def explore_schema():
    '''
    Print header with some example values.
    '''
    for filename in ROW_COUNTS.iterkeys():
        print '-' * 80
        print filename
        for header, row in load_rows(filename):
            for name, value in sorted(zip(header, row)):
                print name.rjust(20), value
            break


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


DATATYPES = {
    'count': {'model': 'gp', 'transform': transform_count},
    'real': {'model': 'nich', 'transform': transform_real},
    'date': {'model': 'nich', 'transform': transform_days_ago},
    'percent': {'model': 'nich', 'transform': transform_percent},
    'categorical': {'model': 'dd', 'transform': transform_string},
    'unbounded_categorical': {'model': 'dpd', 'transform': transform_string},
    'maybe_count': {
        'parts': ['_nonzero', '_value'],
        'model': ['bb', 'gp'],
        'transform': transform_maybe_count,
    },
    'sparse_real': {
        'parts': ['_nonzero', '_value'],
        'model': ['bb', 'nich'],
        'transform': transform_sparse_real,
    },
    'text': None,  # TODO split text into keywords
}


SCHEMA_IN = {
    'acc_now_delinq': 'count',
    'acc_open_past_24mths': 'count',
    'accept_d': 'date',
    'addr_city': 'unbounded_categorica',
    'addr_state': 'unbounded_categorical',
    'annual_inc': 'real',
    'avg_cur_bal': 'real',
    'bc_open_to_buy': 'real',
    'bc_util': 'real',
    'chargeoff_within_12_mths': 'count',
    'collection_recovery_fee': 'sparse_real',
    'collections_12_mths_ex_med': 'count',
    'delinq_2yrs': 'count',
    'delinq_amnt': 'count',
    'desc': 'text',
    'dti': 'real',
    'earliest_cr_line': 'date',
    'emp_length': 'categorical',
    'emp_title': 'unbounded_categorical',
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
    'num_accts_ever_120_pd': 'date',
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
    'purpose': 'text',
    'pymnt_plan': 'categorical',
    'recoveries': 'sparse_real',
    'revol_bal': 'real',
    'revol_util': 'percent',
    'sub_grade': 'categorical',
    'tax_liens': 'count',
    'term': 'categorical',
    'title': 'unbounded_categorical',
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
    for key, name in SCHEMA_IN.iteritems():
        if name is not None:
            datatype = DATATYPES[name]
            model = datatype['model']
            if 'parts' in datatype:
                for part, model_part in izip(datatype['parts'], model):
                    schema[key + part] = model_part
            else:
                schema[key] = model
    return schema



def transform_header():
    header = []
    for key, name in SCHEMA_IN.iteritems():
        if name is not None:
            datatype = DATATYPES[name]
            if 'parts' in datatype:
                for part in datatype['parts']:
                    header.append(key + part)
            else:
                header.append(key)
    return header



def transform_row(row_in):
    row_out = {}
    for key, value in row_in:
        name = SCHEMA_IN[key]
        if name is not None:
            datatype = DATATYPES[name]
            value = datatype['transform'](value)
            if 'parts' in datatype:
                for part, value_part in izip(datatype['parts'], value):
                    row_out[key + part] = value_part
            else:
                row_out[key] = value


def transform_rows(filename):
    header = transform_header()
    filename_out = os.path.join(ROWS_CSV, filename + '.gz')
    with open_compressed(filename_out, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for header_in, row_in in load_rows(filename):
            row_out = transform_row(izip(header_in, row_in))
            writer.writerow(row_out[key] for key in header)


@parsable.command
def transform():
    '''
    Transform dataset.
    '''
    schema = transform_schema()
    json_dump(schema, SCHEMA_JSON)
    loom.util.mkdir_p(ROWS_CSV)
    for filename in ROW_COUNTS.iterkeys():
        transform_rows(filename)


@parsable.command
def ingest():
    '''
    Ingest dataset into loom.
    '''
    loom.tasks.ingest('lending-club', SCHEMA_JSON, ROWS_CSV)


@parsable.command
def infer():
    '''
    Infer model.
    '''
    loom.tasks.infer('lending-club')


if __name__ == '__main__':
    parsable.dispatch()
