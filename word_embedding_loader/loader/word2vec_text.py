# -*- coding: utf-8 -*-
"""
Low level API for loading of word embedding file that was implemented in
`word2vec <https://code.google.com/archive/p/word2vec/>`_, by Mikolov.
This implementation is for word embedding file created with ``-binary 0``
option (the default).
"""

from __future__ import absolute_import, division, print_function, \
    unicode_literals

import numpy as np

from word_embedding_loader import ParseError, parse_warn


def check_valid(line0, line1):
    """
    Check :func:`word_embedding_loader.loader.glove.check_valid` for the API.
    """
    data0 = line0.split(b' ')
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
    # Split and cast a line
    try:
        data = line.strip().split(b' ')
        token = data[0]
        v = list(map(dtype, data[1:]))
    except (ValueError, IndexError):
        raise ParseError(b'Parsing error in line: ' + line)
    return token, v


def _load_line(line, dtype, size):
    # Parse line and do a sanity check
    token, v = _parse_line(line, dtype)
    if len(v) != size:
        raise ParseError(b'Vector size did not match in line: ' + line)
    return token, v


def load_with_vocab(fin, vocab, dtype=np.float32):
    """
    Refer to :func:`word_embedding_loader.loader.glove.load_with_vocab` for the API.
    """
    line = next(fin)
    data = line.strip().split(b' ')
    assert len(data) == 2
    size = int(data[1])
    arr = np.empty((len(vocab), size), dtype=dtype)
    arr.fill(np.NaN)
    for line in fin:
        token, v = _load_line(line, dtype, size)
        if token in vocab:
            arr[vocab[token], :] = v
    if np.any(np.isnan(arr)):
        raise ParseError(b"Some of vocab was not found in word embedding file")
    return arr


def load(fin, dtype=np.float32, max_vocab=None):
    """
    Refer to :func:`word_embedding_loader.loader.glove.load` for the API.
    """
    vocab = {}
    line = next(fin)
    data = line.strip().split(b' ')
    assert len(data) == 2
    words = int(data[0])
    if max_vocab is not None:
        words = min(max_vocab, words)
    size = int(data[1])
    arr = np.empty((words, size), dtype=dtype)
    i = 0
    for n_line, line in enumerate(fin):
        if i >= words:
            break
        token, v = _load_line(line, dtype, size)
        if token in vocab:
            parse_warn(b'Duplicated vocabulary ' + token)
            continue
        arr[i, :] = v
        vocab[token] = i
        i += 1
    if i != words:
        # Use + instead of formatting because python 3.4.* does not allow
        # format with bytes
        parse_warn(
            b'EOF before the defined size (read ' + bytes(i) + b', expected '
            + bytes(words) + b')'
        )
        arr = arr[:i, :]
    return arr, vocab
