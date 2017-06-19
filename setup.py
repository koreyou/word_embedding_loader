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
    maintainer='Yuta Koreeda',
    version=version,
    description='Loaders and savers for different implentations of word embedding.',
    long_description=readme,
    url='https://github.com/koreyou/word_embedding_loader',
    packages=['word_embedding_loader', ],
    license='MIT',
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
