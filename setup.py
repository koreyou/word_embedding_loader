#from Cython.Distutils import build_ext
from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy

ext_modules = [
    Extension(
        "word_embedding_loader.loader.word2vec_bin",
        ["word_embedding_loader/loader/word2vec_bin.pyx"],
        include_dirs=[numpy.get_include()]
    )
]

setup(
    name='WordEmbeddingLoader',
    version='0.1',
    packages=['word_embedding_loader', ],
    license='Creative Commons BY',
#    cmdclass = {'build_ext': build_ext},
    ext_modules=cythonize(ext_modules),
    setup_requires = ['pytest-runner', ],
    tests_require = ['pytest', 'pytest-cov']
)
