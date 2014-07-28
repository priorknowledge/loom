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
import csv
import urllib
import parsable
from distributions.io.stream import (
    open_compressed,
    protobuf_stream_write,
    protobuf_stream_dump,
)
import loom.datasets
import loom.tasks
from loom.schema_pb2 import ProductValue, Row

UCI = 'https://archive.ics.uci.edu/ml/machine-learning-databases/bag-of-words/'
DATA_URL = UCI + 'docword.pubmed.txt.gz'
VOCAB_URL = UCI + 'vocab.pubmed.txt'

NAME = 'pubmed'
DATA = os.path.join(os.path.dirname(__file__), 'data')
RAW = os.path.join(DATA, os.path.basename(DATA_URL))
VOCAB = os.path.join(DATA, os.path.basename(VOCAB_URL))
SCHEMA_CSV = os.path.join(DATA, 'schema.csv.gz')
ROWS_CSV = os.path.join(DATA, 'rows.csv.gz')
SCHEMA = os.path.join(DATA, 'schema.json.gz')
TARES = os.path.join(DATA, 'tares.pbs.gz')
DIFFS = os.path.join(DATA, 'diffs.pbs.gz')

dot_counter = 0


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
    Download datset from UCI website.
    '''
    if not os.path.exists(DATA):
        os.makedirs(DATA)
    if not os.path.exists(RAW):
        print 'fetching {}'.format(DATA_URL)
        urllib.urlretrieve(DATA_URL, RAW)
    if not os.path.exists(VOCAB):
        print 'fetching {}'.format(VOCAB_URL)
        urllib.urlretrieve(VOCAB_URL, VOCAB)


def import_schema():
    schema = [line.strip() for line in open_compressed(VOCAB)]
    with open_compressed(SCHEMA_CSV, 'w') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(schema)
    return schema


def import_tares(schema):
    tare = ProductValue()
    tare.observed.sparsity = ProductValue.Observed.ALL
    tare.booleans[:] = [False] * len(schema)
    protobuf_stream_dump([tare.SerializeToString()], TARES)


def import_rows():
    row = Row()
    pos = row.diff.pos
    neg = row.diff.neg
    pos.observed.sparsity = ProductValue.Observed.SPARSE
    neg.observed.sparsity = ProductValue.Observed.SPARSE
    with open_compressed(RAW) as infile:
        doc_count = int(infile.next())
        word_count = int(infile.next())
        observed_count = int(infile.next())
        print 'Importing {} observations of {} words in {} documents'.format(
            observed_count,
            word_count,
            doc_count)
        with open_compressed(DIFFS, 'wb') as outfile:
            current_doc = None
            for line in infile:
                doc, feature, count = line.split()
                if doc != current_doc:
                    if current_doc is not None:
                        pos.observed.sparse.sort()
                        neg.observed.sparse.sort()
                        protobuf_stream_write(row.SerializeToString(), outfile)
                        print_dot(every=1000)
                    current_doc = doc
                    row.id = int(doc)
                    del pos.booleans[:]
                    del pos.observed.sparse[:]
                    del neg.booleans[:]
                    del neg.observed.sparse[:]
                feature = int(feature) - 1
                pos.observed.sparse.append(feature)
                pos.booleans.append(True)
                neg.observed.sparse.append(feature)
                neg.booleans.append(False)
            protobuf_stream_write(row.SerializeToString(), outfile)


@parsable.command
def ingest():
    '''
    Ingest dataset.
    '''
    schema = import_schema()
    import_tares(schema)
    import_rows()
    raise NotImplementedError('load dataset from tares and diffs')
    loom.datasets.load(NAME, ROWS_CSV, SCHEMA)
    loom.tasks.ingest(NAME)


@parsable.command
def infer(sample_count=10):
    '''
    Infer samples from data.
    '''
    loom.tasks.infer(NAME, sample_count=sample_count)


if __name__ == '__main__':
    parsable.dispatch()
