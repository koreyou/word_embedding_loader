# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import numpy as np


def check_valid(line0, line1):
    data = line0.strip().split(u' ')
    if len(data) <= 2:
        return False
    # check if data[2:] is float values
    try:
        map(float, data[2:])
    except:
        return False
    return True


def _parse_line(line, dtype):
    data = line.strip().split(u' ')
    token = data[0]
    v = map(dtype, data[1:])
    return token, v


def load(fin, vocab_list=None, dtype=np.float32):
    vocab = {}
    arr = None
    i = 0
    for line in fin:
        token, v = _parse_line(line, dtype)
        if vocab_list is not None and token not in vocab_list:
            continue
        if arr is None:
            arr = np.array(v, dtype=dtype).reshape(1, -1)
        else:
            arr = np.append(arr, [v], axis=0)
        vocab[token] = i
        i += 1
    ranks = np.arange(len(arr))
    return arr, vocab, None, ranks
