import os, sys

from distutils.core import setup, Extension
from distutils import sysconfig

cpp_args = ['-std=c++11', '-stdlib=libc++', '-mmacosx-version-min=10.7']

ext_modules = [
    Extension(
    'example',
        ['example.cpp'],
        include_dirs=['pybind11/include', '/Users/azzeddinetiba/cpp_lib/eigen-3.4.0'],
    language='c++',
    extra_compile_args = cpp_args,
    ),
]

setup(
    name='example',
    version='0.0.1',
    description='Example',
    ext_modules=ext_modules,
)