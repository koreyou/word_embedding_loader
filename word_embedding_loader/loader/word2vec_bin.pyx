from collections import OrderedDict

import ctypes
from libc.stdio cimport FILE, fscanf, fread, fdopen
import numpy as np
cimport numpy as np
from cpython cimport bool

ctypedef np.float32_t FLOAT


def check_valid(line0, line1):
    # Only check the first line
    data0 = line0.split(u' ')
    if len(data0) != 2:
        return False
    # check if data0 is int values
    try:
        map(int, data0)
    except:
        return False
    return True


cdef _load_impl(FILE *f, long long words, long long size, bool keep_order,
               set vocab_list, str encoding, bool is_encoded, str errors):
    cdef char ch
    cdef int l
    cdef char[100] vocab
    vocabs = OrderedDict() if keep_order else dict()
    cdef np.ndarray[FLOAT, ndim=2, mode="c"] arr = np.zeros([words, size], dtype=np.float32)
    cdef int i = 0
    while True:
        if i >= words:
            break
        fscanf(f, "%s%n%c", &vocab, &l, &ch)
        if is_encoded:
            ustring = vocab[:l - 1].decode(encoding, errors=errors)
            vocabs[ustring] = i
        else:
            vocabs[<bytes>vocab[:l - 1]] = i
        fread(&arr[i, 0], sizeof(FLOAT), size, f)
        i += 1
    return arr, vocabs


def load(fin, vocab_list=None, dtype=np.float32, keep_order=False, max_vocab=None,
         encoding='utf-8', errors='strict'):
    cdef FILE *f = fdopen(fin.fileno(), 'rb') # attach the stream
    if (f) == NULL:
       raise IOError()
    cdef long long words, size
    fscanf(f, '%lld', &words)
    fscanf(f, '%lld', &size)
    if max_vocab is None:
        words = words
    else:
        words = min(max_vocab, words)
    ret = _load_impl(f, words, size, keep_order, vocab_list, encoding,
                     encoding is not None, errors)
    arr, vocabs = ret
    return arr.astype(dtype), vocabs
