# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

from collections import OrderedDict

import numpy as np

from word_embedding_loader import ParseError, parse_warn


def check_valid(line0, line1):
    data = line0.strip().split(' ')
    if len(data) <= 2:
        return False
    # check if data[2:] is float values
    try:
        map(float, data[2:])
    except:
        return False
    return True


def _parse_line(line, dtype):
    data = line.strip().split(' ')
    token = data[0]
    v = map(dtype, data[1:])
    return token, v


def load(fin, dtype=np.float32, keep_order=False, max_vocab=None,
         encoding='utf-8'):
    vocab = OrderedDict() if keep_order else dict()
    arr = None
    i = 0
    for line in fin:
        if max_vocab is not None and i >= max_vocab:
            break
        try:
            token, v = _parse_line(line, dtype)
        except (ValueError, IndexError):
            raise ParseError('Parsing error in line: %s' % line)
        if token in vocab:
            parse_warn('Duplicated vocabulary %s' % token.encode(encoding))
            continue
        if arr is None:
            arr = np.array(v, dtype=dtype).reshape(1, -1)
        else:
            if arr.shape[1] != len(v):
                raise ParseError('Vector size did not match in line: %s' % line)
            arr = np.append(arr, [v], axis=0)

        if encoding is not None:
            token = token.decode(encoding)
        vocab[token] = i
        i += 1
    return arr, vocab
