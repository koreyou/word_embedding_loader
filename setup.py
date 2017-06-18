import os

from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize
import sphinx.apidoc
from sphinx.setup_command import BuildDoc
import numpy

ext_modules = [
    Extension(
        "word_embedding_loader.loader.word2vec_bin",
        ["word_embedding_loader/loader/word2vec_bin.pyx"],
        include_dirs=[numpy.get_include()]
    ),
    Extension(
        "word_embedding_loader.saver.word2vec_bin",
        ["word_embedding_loader/saver/word2vec_bin.pyx"],
        include_dirs=[numpy.get_include()], language="c++"
    )
]

cmdclass = {}

for e in ext_modules:
    e.cython_directives = {"embedsignature": True}


try:
    from sphinx.setup_command import BuildDoc
    cmdclass['build_sphinx'] = BuildDoc
except ImportError:
    pass


name = 'WordEmbeddingLoader'
version = '0.1'
release = '0.1.0'

setup(
    name=name,
    author='Yuta Koreeda',
    version=version,
    packages=['word_embedding_loader', ],
    license='Creative Commons BY',
    cmdclass = cmdclass,
    install_requires=[
        'Click',
    ],
    entry_points = {
        'console_scripts': ['word-embedding-loader=word_embedding_loader.cli:cli'],
    },
    command_options={
        'build_sphinx': {
            'project': ('setup.py', name),
            'version': ('setup.py', version),
            'release': ('setup.py', release)}},
    ext_modules=cythonize(ext_modules),
    setup_requires = ['pytest-runner',
                      'sphinx',
                      'sphinxcontrib-napoleon',
                      'sphinx_rtd_theme'],
    tests_require = ['pytest', 'pytest-cov']
)
