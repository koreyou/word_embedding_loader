import os

from setuptools import setup
from setuptools.extension import Extension
from setuptools.command.build_py import build_py


cython_modules = [
    ["word_embedding_loader", "loader", "word2vec_bin"],
    ["word_embedding_loader", "saver", "word2vec_bin"]
]

class BuildPyCommand(build_py):
    """ Custom build routine """
    def run(self):
        import numpy
        from Cython.Build import cythonize
        ext_modules = [
            Extension(
                '.'.join(mod), ['/'.join(mod) + '.pyx'],
                include_dirs=[numpy.get_include()], language="c++"
            ) for mod in cython_modules
        ]
        # Add signiture for Sphinx
        for e in ext_modules:
            e.cython_directives = {"embedsignature": True}

        cythonize(ext_modules)
        build_py.run(self)


name = 'WordEmbeddingLoader'
version = '0.1'
release = '0.1.0'

setup(
    name=name,
    author='Yuta Koreeda',
    version=version,
    packages=['word_embedding_loader', ],
    license='MIT License',
    cmdclass = {'build_py': BuildPyCommand},
    install_requires=[
        'Click',
        'numpy'
    ],
    entry_points = {
        'console_scripts': ['word-embedding-loader=word_embedding_loader.cli:cli'],
    },
    command_options={
        'build_sphinx': {
            'project': ('setup.py', name),
            'version': ('setup.py', version),
            'release': ('setup.py', release)}},
    setup_requires = ['pytest-runner',
                      'sphinx',
                      'sphinxcontrib-napoleon',
                      'sphinx_rtd_theme',
                      'numpy',
                      'Cython'
                      ],
    tests_require = ['pytest', 'pytest-cov', 'Cython']
)
