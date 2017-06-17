# -*- coding: utf-8 -*-
u"""

Low level API for loading of word embedding file that was implemented in
`word2vec <https://code.google.com/archive/p/word2vec/>`_, by Mikolov.
This implementation is for word embedding file created with ``-binary 1``
option.
"""

from collections import OrderedDict

import ctypes
from libc.stdio cimport FILE, fprintf, fwrite, fdopen, fflush, ferror
import numpy as np
cimport numpy as np
from libcpp.pair cimport pair
from libcpp.vector cimport vector
from libcpp.string cimport string
from cython.operator cimport dereference, preincrement

ctypedef np.float32_t FLOAT


cdef int _save_impl(FILE *f, np.ndarray[FLOAT, ndim=2, mode="c"] arr, vector[pair[string, int]]& vocab, long long size) except -1:
    cdef vector[pair[string, int]].iterator end = vocab.end()
    cdef vector[pair[string, int]].iterator it = vocab.begin()
    cdef int ret = 0

    while it != end:
        # Write word
        fprintf(f, "\n%s ", dereference(it).first.c_str())
        if ferror(f) != 0:
            ret = -1
            break
        # Write the vector
        fwrite(&arr[dereference(it).second, 0], sizeof(FLOAT), size, f)
        if ferror(f) != 0:
            ret = -1
            break
        preincrement(it)
    return ret


def _mapper(encoding, errors):
    def body(item):
        key, value = item
        return (key.encode(encoding, errors=errors), value)
    return body


def _count_sorter(counts):
    def body(item):
        key, value = item
        return counts(key)
    return body


def _value_sort(item):
    key, value = item
    return value


def save(f, arr, vocab, counts=None, encoding='utf-8', unicode_errors='strict'):
    u"""
    Refer to :func:`word_embedding_loader.saver.glove.save` for the API.
    """
    vocab = map(_mapper(encoding, unicode_errors), vocab.iteritems())
    if counts is None:
        if isinstance(vocab, OrderedDict):
            itr = vocab
        else:
            itr = sorted(vocab, key=_value_sort)
    else:
        itr = sorted(vocab, key=_count_sorter(counts), reverse=True)

    cdef long long size = arr.shape[1]
    cdef long long words = arr.shape[0]
    cdef np.ndarray[FLOAT, ndim=2, mode="c"] carr = arr
    cdef vector[pair[string, int]] cvocab = itr

    cdef FILE *fin = fdopen(f.fileno(), 'w') # attach the stream
    if (fin) == NULL:
       raise IOError()
    cdef int ret = 0
    ret = fprintf(fin, '%lld %lld', words, size)
    if ret < 0:
        raise IOError()
    ret = _save_impl(fin, carr, cvocab, size)
    if ret < 0:
        raise IOError()
    fflush(fin)
    if ferror(fin) != 0:
        raise IOError()
