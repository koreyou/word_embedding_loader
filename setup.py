import os

from setuptools import setup
from setuptools.extension import Extension
from setuptools.command.build_ext import build_ext as _build_ext



cython_modules = [
    ["word_embedding_loader", "loader", "word2vec_bin"],
    ["word_embedding_loader", "saver", "word2vec_bin"]
]

ext_modules = [
    Extension(
        '.'.join(mod), ['/'.join(mod) + '.cpp'],
        language="c++"
    ) for mod in cython_modules
]

class BuildPyCommand(_build_ext):
    """ Custom build routine """
#    def run(self):

    def run(self):
        import numpy
        from Cython.Build import cythonize
        _ext_modules = [Extension(
            '.'.join(mod), ['/'.join(mod) + '.cpp'],
            language="c++", include_dirs=[numpy.get_include()]
        ) for mod in cython_modules]
        for i in xrange(len(self.extensions)):
            self.extensions[i].include_dirs.append(numpy.get_include())
            # Add signiture for Sphinx
            self.extensions[i].cython_directives = {"embedsignature": True}
        cythonize(_ext_modules)
        _build_ext.run(self)

try:
    with open('README.rst') as f:
        readme = f.read()
except IOError:
    readme = ''


name = 'WordEmbeddingLoader'
version = '0.1'
release = '0.1.1'

setup(
    name=name,
    author='Yuta Koreeda',
    author_email='secret-email@example.com',
    maintainer='Yuta Koreeda',
    maintainer_email='secret-email@example.com',
    version=release,
    description='Loaders and savers for different implentations of word embedding.',
    long_description=readme,
    url='https://github.com/koreyou/word_embedding_loader',
    packages=['word_embedding_loader',
              'word_embedding_loader.loader',
              'word_embedding_loader.saver'
              ],
    ext_modules=ext_modules,
    license='MIT',
    cmdclass = {'build_ext': BuildPyCommand},
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
    tests_require = ['pytest', 'pytest-cov', 'Cython'],
    classifiers=[
        "Environment :: Console",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Cython",
        "Topic :: Documentation :: Sphinx",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis"
    ]
)
