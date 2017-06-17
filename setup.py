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

class BuildDocApiDoc(BuildDoc, object):
    # inherit from object to enable 'super'
    user_options = []
    description = 'sphinx'
    def run(self):
        # metadata contains information supplied in setup()
        metadata = self.distribution.metadata
        # package_dir may be None, in that case use the current directory.
        src_dir = (self.distribution.package_dir or {'': ''})['']
        src_dir = os.path.join(os.getcwd(),  src_dir)
        # Run sphinx by calling the main method, '--full' also adds a conf.py
        sphinx.apidoc.main(
            ['', '-f', '-H', metadata.name, '-A', metadata.author,
             '-V', metadata.version, '-R', metadata.version, '-T', '-M',
             '-o', os.path.join('doc', 'source', 'modules'), src_dir])
        super(BuildDocApiDoc, self).run()


name = 'WordEmbeddingLoader'
version = '0.1'
release = '0.1.0'

setup(
    name=name,
    author='Yuta Koreeda',
    version=version,
    packages=['word_embedding_loader', ],
    license='Creative Commons BY',
    # cmdclass = {'build_ext': build_ext},
    cmdclass = {'build_sphinx': BuildDocApiDoc},
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
