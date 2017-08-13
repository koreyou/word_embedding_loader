# -*- coding: utf-8 -*-
"""
Low level API for loading of word embedding file that was implemented in
`GloVe <https://nlp.stanford.edu/projects/glove/>`_, Global Vectors for Word
Representation, by Jeffrey Pennington, Richard Socher, Christopher D. Manning
from Stanford NLP group.
"""

from __future__ import absolute_import, division, print_function, \
    unicode_literals

import numpy as np

from word_embedding_loader import ParseError, parse_warn


def check_valid(line0, line1):
    """
    Check if a file is valid Glove format.

    Args:
        line0 (bytes): First line of the file
        line1 (bytes): Second line of the file

    Returns:
        boo: ``True`` if it is valid. ``False`` if it is invalid.

    """
    data = line0.strip().split(b' ')
    if len(data) <= 2:
        return False
    # check if data[2:] is float values
    try:
        map(float, data[2:])
    except:
        return False
    return True


def _parse_line(line, dtype):
    data = line.strip().split(b' ')
    token = data[0]
    v = list(map(dtype, data[1:]))
    return token, v


def load_with_vocab(fin, vocab, dtype=np.float32):
    """
    Load word embedding file with predefined vocabulary

    Args:
        fin (File): File object to read. File should be open for reading ascii.
        vocab (dict): Mapping from words (``bytes``) to vector indices
            (``int``).
        dtype (numpy.dtype): Element data type to use for the array.

    Returns:
        numpy.ndarray: Word embedding representation vectors
    """
    arr = None
    for line in fin:
        try:
            token, v = _parse_line(line, dtype)
        except (ValueError, IndexError):
            raise ParseError(b'Parsing error in line: ' + line)
        if token in vocab:
            if arr is None:
                arr = np.empty((len(vocab), len(v)), dtype=dtype)
                arr.fill(np.NaN)
            elif arr.shape[1] != len(v):
                raise ParseError(b'Vector size did not match in line: ' + line)
            arr[vocab[token], :] = np.array(v, dtype=dtype).reshape(1, -1)
    return arr


def load(fin, dtype=np.float32, max_vocab=None):
    """
    Load word embedding file.

    Args:
        fin (File): File object to read. File should be open for reading ascii.
        dtype (numpy.dtype): Element data type to use for the array.
        max_vocab (int): Number of vocabulary to read.

    Returns:
        numpy.ndarray: Word embedding representation vectors
        dict: Mapping from words to vector indices.

    """
    vocab = {}
    arr = None
    i = 0
    for line in fin:
        if max_vocab is not None and i >= max_vocab:
            break
        try:
            token, v = _parse_line(line, dtype)
        except (ValueError, IndexError):
            raise ParseError(b'Parsing error in line: ' + line)
        if token in vocab:
            parse_warn(b'Duplicated vocabulary ' + token)
            continue
        if arr is None:
            arr = np.array(v, dtype=dtype).reshape(1, -1)
        else:
            if arr.shape[1] != len(v):
                raise ParseError(b'Vector size did not match in line: ' + line)
            arr = np.append(arr, [v], axis=0)
        vocab[token] = i
        i += 1
    return arr, vocab
