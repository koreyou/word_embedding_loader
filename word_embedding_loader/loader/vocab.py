# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, \
    unicode_literals

from collections import OrderedDict


def load_vocab(fin):
    """
    Load vocabulary from vocab file created by word2vec with
    ``-save-vocab <file>`` option.

    Args:
        fin (File): File-like object to read from.
        encoding (bytes): Encoding of the input file as defined in ``codecs``
            module of Python standard library.
        errors (bytes): Set the error handling scheme. The default error
            handler is 'strict' meaning that encoding errors raise ValueError.
            Refer to ``codecs`` module for more information.

    Returns:
        OrderedDict: Mapping from a word (``bytes``) to the number of
            appearance in the original text (``int``). Order are preserved from
            the original vocab file.
    """
    vocab = OrderedDict()
    for line in fin:
        v, c = line.strip().split()
        vocab[v] = int(c)
    return vocab
