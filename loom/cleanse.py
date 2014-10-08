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
import itertools
from contextlib2 import ExitStack
from distributions.io.stream import open_compressed
import loom.util

DEFAULT_PART_COUNT = 100
re_nonascii = re.compile('[^\x00-\x7f]+')


def force_ascii(filename_in, filename_out=None, size=4096):
    with ExitStack() as stack:
        with_ = stack.enter_context
        if filename_out is None:
            filename_out = with_(loom.util.temp_copy(filename_in))
        source = with_(open_compressed(filename_in, 'rb'))
        destin = with_(open_compressed(filename_out, 'w'))
        chunk = source.read(size)
        while chunk:
            destin.write(re_nonascii.sub('', chunk))
            chunk = source.read(size)


def repartition_csv_files(infiles, outfiles):
    with ExitStack() as stack:
        with_ = stack.enter_context
        readers = [with_(loom.util.csv_reader(f)) for f in infiles]
        writers = [with_(loom.util.csv_writer(f)) for f in outfiles]
        headers = [reader.next() for reader in readers]
        header = headers[0]
        assert all(h == header for h in headers), 'headers to not match'
        for writer in writers:
            writer.writerow(header)
        len_header = len(header)
        get_writer = itertools.cycle(writers).next
        for reader in readers:
            for row in reader:
                if row:
                    assert len(row) == len_header, row
                    get_writer().writerow(row)


def repartition_csv_dir(dirname, part_count=DEFAULT_PART_COUNT):
    dirname = os.path.abspath(dirname)
    assert part_count >= 1, part_count
    parts = os.path.basename(min(os.listdir(dirname))).split('.')
    if parts[-1] in ['gz', 'bz2']:
        parts = [parts[0]] + ['{}'] + parts[-2:]
    else:
        parts = [parts[0]] + ['{}'] + parts[-1:]
    with loom.util.temp_copy(dirname) as temp:
        loom.util.mkdir_p(temp)
        outfile = os.path.join(temp, '.'.join(parts)).format
        outfiles = [outfile(i) for i in xrange(part_count)]
        infiles = [os.path.join(dirname, f) for f in os.listdir(dirname)]
        repartition_csv_files(infiles, outfiles)
        loom.util.rm_rf(dirname)  # HACK not atomic
