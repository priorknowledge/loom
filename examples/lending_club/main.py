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
import subprocess
import urllib
import loom.util
import parsable

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
ROWS_CSV = os.ph.join(DATA, 'rows_csv')


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


transforms = {
    'term': lambda x: x.replace('months', '').strip(),
}


@parsable.command
def ingest():
    loom.util.mkdir_p(ROWS_CSV)
    for filename in FILES:
        loom.util.cp_ns(
            os.path.join(DOWNLOADS, filename),
            os.path.join(ROWS_CSV, filename))
    raise NotImplementedError('TODO transform features')

if __name__ == '__main__':
    parsable.dispatch()
