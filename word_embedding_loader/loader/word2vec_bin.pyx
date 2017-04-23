
import ctypes
from libc.stdio cimport FILE, fscanf, fread, fdopen
import numpy as np
cimport numpy as np


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


def load(fin, int max_vocabs):
    cdef FILE *f = fdopen(fin.fileno(), 'rb') # attach the stream
    if (f) == NULL:
       raise IOError()
    cdef long long words, size
    cdef char ch
    cdef int l
    cdef char[100] vocab
    vocabs = {}
    fscanf(f, '%lld', &words)
    fscanf(f, '%lld', &size)
    size = min(max_vocabs, size)
    cdef np.ndarray[FLOAT, ndim=2, mode="c"] arr = np.zeros([words, size], dtype=np.float32)
    cdef int i = 0
    while True:
        if i >= words:
            break
        fscanf(f, "%s%n%c", &vocab, &l, &ch)
        vocabs[<bytes>vocab[:l - 1]] = i
        fread(&arr[i, 0], sizeof(FLOAT), size, f)
        i += 1
    ranks = np.arange(len(arr))
    return arr, vocabs, None, ranks
