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
from distutils.core import setup, Extension
from Cython.Build import cythonize


library_dirs = []
libraries = ['protobuf', 'distributions_shared']
include_dirs = ['include']
ve = os.environ.get('VIRTUAL_ENV')
if ve:
    include_dirs.append(os.path.join(ve, 'include'))
    library_dirs.append(os.path.join(ve, 'lib'))
extra_compile_args = [
    '-DDIST_DEBUG_LEVEL=3',
    '-DDIST_THROW_ON_ERROR=1',
    '-DLOOM_DEBUG_LEVEL=3',
    '-std=c++0x',
    '-Wall',
    '-Werror',
    '-Wno-unused-function',
    '-Wno-sign-compare',
    '-Wno-strict-aliasing',
    '-O3',
]


def make_extension(name, sources=[]):
    module = 'loom.' + name
    sources.append('{}.{}'.format(module.replace('.', '/'), 'pyx'))
    return Extension(
        module,
        sources=sources,
        language='c++',
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        extra_compile_args=extra_compile_args,
    )


ext_modules = [
    make_extension('cFormat', sources=['src/schema.pb.cc']),
]


version = None
with open(os.path.join('loom', '__init__.py')) as f:
    for line in f:
        if re.match("__version__ = '\S+'$", line):
            version = line.split()[-1].strip("'")
assert version, 'could not determine version'


with open('README.md') as f:
    long_description = f.read()


config = {
    'version': version,
    'name': 'loom',
    'description': 'Streaming cross-cat inference engine',
    'long_description': long_description,
    'url': 'https://github.com/priorknowledge/loom',
    'author': 'Fritz Obermeyer, Jonathan Glidden',
    'maintainer': 'Fritz Obermeyer',
    'maintainer_email': 'fritz.obermeyer@gmail.com',
    'license': 'Revised BSD',
    'packages': [
        'loom',
        'loom.test',
    ],
    'ext_modules': cythonize(ext_modules),
}

setup(**config)
