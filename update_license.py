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

from __future__ import print_function
import fnmatch
import os
import re
import parsable

SYMBOLS = {
    '//': ['.c', '.cc', '.cpp', '.h', '.hpp', '.proto'],
    '#': ['.py', '.pxd', '.pyx'],
    '%': ['.tex'],
}
extensions = sum(SYMBOLS.values(), [])
SYMBOL_OF = {
    extension: symbol
    for symbol, extensions in SYMBOLS.iteritems()
    for extension in extensions
}

dir_blacklist = [
    '.git',
    'doc',
    'include',  # just an alias for src
    'build',
    'data',
]

file_blacklist = [
    '*.pb.h',
    '*.pb.cc',
    '*_pb2.py',
]

FILES = sorted(
    os.path.join(root, filename)
    for root, dirnames, filenames in os.walk('.')
    if not any(d in root.split('/') for d in dir_blacklist)
    for extension in extensions
    for filename in fnmatch.filter(filenames, '*' + extension)
    if not any(fnmatch.fnmatch(filename, patt) for patt in file_blacklist)
)

LICENSE = []
with open('LICENSE.txt') as f:
    for line in f:
        LICENSE.append(line.rstrip())

HEADERS = {
    symbol: [symbol + ' ' + line if line else symbol for line in LICENSE]
    for symbol in SYMBOLS
}


@parsable.command
def show():
    '''
    List all files that should have a license.
    '''
    for filename in FILES:
        print(filename)


def read_and_strip_lines(filename):
    extension = re.search('\.[^.]*$', filename).group()
    symbol = SYMBOL_OF[extension]
    lines = []
    with open(filename) as i:
        writing = False
        for line in i:
            line = line.rstrip()
            if not writing and line and not line.startswith(symbol):
                writing = True
            if writing:
                lines.append(line)
    return lines


def write_lines(lines, filename):
    with open(filename, 'w') as f:
        for line in lines:
            print(line, file=f)


@parsable.command
def strip():
    '''
    Strip headers from all files.
    '''
    for filename in FILES:
        lines = read_and_strip_lines(filename)
        write_lines(lines, filename)


@parsable.command
def update():
    '''
    Update headers on all files to match LICNESE.txt.
    '''
    for filename in FILES:
        extension = re.search('\.[^.]*$', filename).group()
        symbol = SYMBOL_OF[extension]
        lines = read_and_strip_lines(filename)
        if lines and lines[0]:
            if extension == '.py' and lines[0].startswith('class '):
                lines = [''] + lines  # pep8 compliance
            write_lines(HEADERS[symbol] + [''] + lines, filename)


if __name__ == '__main__':
    parsable.dispatch()
