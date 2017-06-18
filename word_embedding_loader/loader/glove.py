# -*- coding: utf-8 -*-
u"""
Low level API for loading of word embedding file that was implemented in
`GloVe <https://nlp.stanford.edu/projects/glove/>`_, Global Vectors for Word
Representation, by Jeffrey Pennington, Richard Socher, Christopher D. Manning
from Stanford NLP group.
"""

from __future__ import absolute_import, print_function

import numpy as np

from word_embedding_loader import ParseError, parse_warn


def check_valid(line0, line1):
    """
    Check if a file is valid Glove format.

    Args:
        line0 (str): First line of the file
        line1 (str): Second line of the file

    Returns:
        boo: ``True`` if it is valid. ``False`` if it is invalid.

    """
    data = line0.strip().split(' ')
    if len(data) <= 2:
        return False
    # check if data[2:] is float values
    try:
        map(float, data[2:])
    except:
        return False
    return True


def _parse_line(line, dtype, encoding, unicode_errors):
    data = line.strip().split(' ')
    token = data[0]
    v = map(dtype, data[1:])
    if encoding is not None:
        token = token.decode(encoding, errors=unicode_errors)
    return token, v


def load_with_vocab(
        fin, vocab, dtype=np.float32, encoding='utf-8', unicode_errors='strict'):
    u"""
    Load word embedding file with predefined vocabulary

    Args:
        fin (File): File object to read. File should be open for reading ascii.
        vocab (dict): Mapping from words (``unicode``) to vector indices
            (``int``).
        dtype (numpy.dtype): Element data type to use for the array.
        encoding (str): Encoding of the input file as defined in ``codecs``
            module of Python standard library.
        unicode_errors (str): Set the error handling scheme. The default error
            handler is 'strict' meaning that encoding errors raise ValueError.
            Refer to ``codecs`` module for more information.

    Returns:
        numpy.ndarray: Word embedding representation vectors
    """
    arr = None
    for line in fin:
        try:
            token, v = _parse_line(line, dtype, encoding, unicode_errors)
        except (ValueError, IndexError):
            raise ParseError('Parsing error in line: %s' % line)
        if token in vocab:
            if arr is None:
                arr = np.empty((len(vocab), len(v)), dtype=dtype)
                arr.fill(np.NaN)
            elif arr.shape[1] != len(v):
                raise ParseError('Vector size did not match in line: %s' % line)
            arr[vocab[token], :] = np.array(v, dtype=dtype).reshape(1, -1)
    return arr


def load(fin, dtype=np.float32, max_vocab=None, encoding='utf-8',
         unicode_errors='strict'):
    u"""
    Load word embedding file.

    Args:
        fin (File): File object to read. File should be open for reading ascii.
        dtype (numpy.dtype): Element data type to use for the array.
        max_vocab (int): Number of vocabulary to read.
        encoding (str): Encoding of the input file as defined in ``codecs``
            module of Python standard library.
        unicode_errors (str): Set the error handling scheme. The default error
            handler is 'strict' meaning that encoding errors raise ValueError.
            Refer to ``codecs`` module for more information.

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
            token, v = _parse_line(line, dtype, encoding, unicode_errors)
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
        vocab[token] = i
        i += 1
    return arr, vocab
