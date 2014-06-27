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
from itertools import izip
import parsable
from nose import SkipTest
from loom.util import mkdir_p, rm_rf, parallel_map
import loom.format
import loom.generate
import loom.config
import loom.runner

ROOT = os.path.dirname(os.path.abspath(__file__))
SCHEMA = os.path.join(ROOT, 'schema.json')
EXAMPLE = os.path.join(ROOT, 'example.csv')

DATA = os.path.join(ROOT, 'data')
ROWS_CSV = os.path.join(DATA, 'rows_csv')
ENCODING = os.path.join(DATA, 'encoding.json')
ROWS = os.path.join(DATA, 'rows.pbs.gz')

SEED = os.path.join(DATA, 'seeds', '{:03d}')
INIT = os.path.join(SEED, 'init.pb.gz')
CONFIG = os.path.join(SEED, 'config.pb.gz')
SHUFFLED = os.path.join(SEED, 'shuffled.pbs.gz')
MODEL = os.path.join(SEED, 'model')
GROUPS = os.path.join(SEED, 'groups')
ASSIGN = os.path.join(SEED, 'assign.pbs.gz')
LOG = os.path.join(SEED, 'log.pbs')


def s3_connect():
    import boto
    assert 'TAXI_BUCKET' in os.environ, 'TAXI_BUCKET is not defined'
    bucket = os.environ['TAXI_BUCKET']
    return boto.connect_s3().get_bucket(bucket)


def s3_get((source, destin)):
    try:
        print 'starting {}'.format(source)
        conn = s3_connect()
        key = conn.get_key(source)
        key.get_contents_to_filename(destin)
        print 'finished {}'.format(source)
    except:
        rm_rf(destin)
        raise


@parsable.command
def download():
    '''
    Download dataset from S3. Recommended for EC2 machines.
    '''
    conn = s3_connect()
    keys = [key.name for key in conn.list('taxi-dataset/partitioned')]
    patt = re.compile(r'.*/full-taxi-\d\d\d\.csv\.gz$')
    keys = [key for key in keys if patt.search(key)]
    assert keys, 'nothing to download'
    files = [os.path.join(ROWS_CSV, os.path.basename(key)) for key in keys]
    tasks = [
        (source, destin)
        for source, destin in izip(keys, files)
        if not os.path.exists(destin)
    ]
    if tasks:
        print 'starting download of {} files'.format(len(tasks))
        mkdir_p(ROWS_CSV)
        parallel_map(s3_get, tasks)
        print 'finished download of {} files'.format(len(keys))


@parsable.command
def ingest():
    '''
    Make encoding and import rows.
    '''
    if not os.path.exists(ROWS_CSV):
        print 'WARNING using example data. Download dataset to use full data.'
        rows_csv = EXAMPLE
    else:
        rows_csv = ROWS_CSV

    mkdir_p(DATA)
    print 'making encoding'
    loom.format.make_encoding(
        schema_in=SCHEMA,
        rows_in=rows_csv,
        encoding_out=ENCODING)
    print 'importing rows'
    loom.format.import_rows(
        encoding_in=ENCODING,
        rows_in=rows_csv,
        rows_out=ROWS)


@parsable.command
def shuffle(sample_count=1):
    '''
    Run shuffle, one sample worker at a time.
    '''
    for seed in xrange(sample_count):
        mkdir_p(SEED.format(seed))

        print 'shuffling rows'
        loom.runner.shuffle(
            rows_in=ROWS,
            rows_out=SHUFFLED.format(seed),
            seed=seed,
            profile='time')


@parsable.command
def infer(sample_count=1, debug=False):
    '''
    Run inference, one sample worker at a time.
    '''
    for seed in xrange(sample_count):
        mkdir_p(SEED.format(seed))

        rows = SHUFFLED.format(seed)
        if not os.path.exists(rows):
            print 'WARNING using un-shuffled rows. Try running shuffle first.'
            rows = ROWS

        print 'generating init'
        loom.generate.generate_init(
            encoding_in=ENCODING,
            model_out=INIT.format(seed),
            seed=seed)

        print 'creating config'
        config = {'seed': seed}
        loom.config.config_dump(config, CONFIG.format(seed))

        print 'inferring, watch {}'.format(LOG.format(seed))
        loom.runner.infer(
            config_in=CONFIG.format(seed),
            rows_in=rows,
            model_in=INIT.format(seed),
            model_out=MODEL.format(seed),
            groups_out=GROUPS.format(seed),
            assign_out=ASSIGN.format(seed),
            log_out=LOG.format(seed),
            debug=debug,
            profile='time')


@parsable.command
def test():
    '''
    Test on tiny dataset, if full dataset has not already been downloaded.
    '''
    if os.path.exists(ROWS_CSV):
        raise SkipTest('avoid testing on large dataset')
    ingest()
    shuffle(sample_count=2)
    infer(sample_count=2, debug=True)


if __name__ == '__main__':
    parsable.dispatch()
