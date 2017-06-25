import os

from setuptools import setup
from setuptools.extension import Extension
from setuptools.command.sdist import sdist as _sdist


cython_modules = [
    ["word_embedding_loader", "loader", "word2vec_bin"],
    ["word_embedding_loader", "saver", "word2vec_bin"]
]


def _cythonize(extensions, apply_cythonize):
    import numpy
    ext = '.pyx' if apply_cythonize else '.cpp'
    extensions = [
        Extension(
            '.'.join(mod), ['/'.join(mod) + ext],
            language="c++"
        ) for mod in extensions
    ]
    for i in xrange(len(extensions)):
        extensions[i].include_dirs.append(numpy.get_include())
        # Add signiture for Sphinx
        extensions[i].cython_directives = {"embedsignature": True}
    if apply_cythonize:
        from Cython.Build import cythonize
        extensions = cythonize(extensions)
    return extensions


class sdist(_sdist):
    def run(self):
        # Force cythonize for sdist
        _cythonize(cython_modules, True)
        _sdist.run(self)


class lazy_cythonize(list):
    # Adopted from https://stackoverflow.com/a/26698408/7820599
    def _cythonize(self):
        self._list = _cythonize(self._list, self._apply_cythonize)
        self._is_cythonized = True

    def __init__(self, extensions, apply_cythonize=False):
        super(lazy_cythonize, self).__init__()
        self._list = extensions
        self._apply_cythonize = apply_cythonize
        self._is_cythonized = False

    def c_list(self):
        if not self._is_cythonized:
            self._cythonize()
        return self._list

    def __iter__(self):
        for e in self.c_list():
            yield e

    def __getitem__(self, ii):
        return self.c_list()[ii]

    def __len__(self):
        return len(self.c_list())


try:
    with open('README.rst') as f:
        readme = f.read()
except IOError:
    readme = ''


name = 'WordEmbeddingLoader'
version = '0.1'
release = '0.1.0'

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
    ext_modules=lazy_cythonize(
        cython_modules,
        os.environ.get('DEVELOP_WE', os.environ.get('READTHEDOCS')) is not None
    ),
    license='MIT',
    cmdclass = {'sdist': sdist},
    install_requires=[
        'Click',
        'numpy>=1.10'
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
                      'numpy>=1.10',
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
