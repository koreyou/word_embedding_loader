# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import numpy as np


def check_valid(line0, line1):
    data0 = line0.split(u' ')
    if len(data0) != 2:
        return False
    # check if data0 is int values
    try:
        map(int, data0)
        _parse_line(line1, float)
    except:
        return False
    return True


def _parse_line(line, dtype):
    data = line.strip().split(' ')
    token = data[0]
    v = map(dtype, data[1:])
    return token, v


def load(fin, vocab_list=None, dtype=np.float32):
    vocab = {}
    line = fin.next()
    data = line.strip().split(u' ')
    assert len(data) == 2
    words = int(data[0])
    size = int(data[1])
    arr = np.empty((words, size), dtype=dtype)
    for i, line in enumerate(fin):
        token, v = _parse_line(line, dtype)
        arr[i, :] = v
        vocab[token] = i
    ranks = np.arange(len(arr))
    return arr, vocab, None, ranks
