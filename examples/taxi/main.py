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
from loom.util import mkdir_p, rm_rf, parallel_map
import loom.format
import loom.datasets
import loom.benchmark

S3_URL = 's3://pk-dsp/taxi-data/partitioned/geocoded'

ROOT = os.path.dirname(os.path.abspath(__file__))
SCHEMA = os.path.join(ROOT, 'schema.json')
EXAMPLE = os.path.join(ROOT, 'example.csv')
ROWS_CSV = os.path.join(ROOT, 'rows_csv')


def s3_split(url):
    bucket, path = re.match(r's3://([^/]*)/(.*)', S3_URL).group(1, 2)
    return bucket, path


def s3_get((bucket, source, destin)):
    import boto
    try:
        print 'starting {}'.format(source)
        conn = boto.connect_s3().get_bucket(bucket)
        key = conn.get_key(source)
        key.get_contents_to_filename(destin)
        print 'finished {}'.format(source)
    except:
        rm_rf(destin)
        raise


@parsable.command
def download(s3_url=S3_URL):
    '''
    Download dataset from S3 and load into loom.benchmark jig.
    '''
    import boto
    bucket, path = s3_split(s3_url)
    conn = boto.connect_s3().get_bucket(bucket)
    keys = [
        key.name
        for key in conn.list(path)
        if re.match(r'.*\d\d\d\.csv\.gz$', key.name)
    ]
    assert keys, 'nothing to download'
    files = [os.path.join(ROWS_CSV, os.path.basename(key)) for key in keys]
    tasks = [
        (bucket, source, destin)
        for source, destin in izip(keys, files)
        if not os.path.exists(destin)
    ]
    if tasks:
        print 'starting download of {} files'.format(len(tasks))
        mkdir_p(ROWS_CSV)
        parallel_map(s3_get, tasks)
        print 'finished download of {} files'.format(len(keys))


@parsable.command
def run():
    '''
    Load; ingest; init; shuffle; infer.
    '''
    name = 'taxi'
    loom.datasets.load(name, SCHEMA, ROWS_CSV)
    loom.benchmark.ingest(name)
    loom.benchmark.init(name)
    loom.benchmark.shuffle(name)
    loom.benchmark.infer(name)


@parsable.command
def test():
    '''
    Test on tiny example dataset.
    '''
    name = 'taxi-test'
    loom.datasets.clean(name)
    loom.datasets.load(name, SCHEMA, EXAMPLE)
    loom.benchmark.ingest(name, profile=None)
    loom.benchmark.init(name)
    loom.benchmark.shuffle(name, profile=None)
    loom.benchmark.infer(name, profile=None)
    loom.benchmark.load_checkpoint(name, period_sec=0.01)
    loom.benchmark.infer_checkpoint(name, profile=None)


if __name__ == '__main__':
    parsable.dispatch()
