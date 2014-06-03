import os
from distutils.core import setup, Extension
from Cython.Build import cythonize
import distributions

libraries = ['protobuf']

include_dirs = [
    'include',
    os.path.join(distributions.ROOT, 'include'),
]

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
        libraries=libraries,
        extra_compile_args=extra_compile_args,
    )


ext_modules = [
    make_extension(
        'cFormat',
        sources=[
            os.path.join(distributions.ROOT, 'src/io/schema.pb.cc'),
            'src/schema.pb.cc',
        ],
    ),
]

config = {
    'name': 'loom',
    'description': 'Streaming cross-cat inference engine',
    'packages': [
        'loom',
        'loom.test',
    ],
    'ext_modules': cythonize(ext_modules),
}

setup(**config)
