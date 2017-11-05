# -*- coding: utf-8 -*-
"""
Low level API for saving of word embedding file to numpy npz format.
"""
from __future__ import absolute_import, division, print_function, \
    unicode_literals

import zipfile

import numpy as np
import six

from word_embedding_loader import util


def check_valid(path):
    """
    Check if a file is valid Glove format.
    It may have side effect to input file-like object

    Args:
        path (str): Path to check validity

    Returns:
        boo: ``True`` if it is valid. ``False`` if it is invalid.

    """
    if not zipfile.is_zipfile(path):
        return False
    files = zipfile.ZipFile(path).namelist()
    return len(files) == 2 and 'arr.npy' in files and 'vocab.npy' in files


def load_with_vocab(fin, vocab, dtype=np.float32):
    """
    Refer to :func:`word_embedding_loader.loader.glove.load` for the API.
    """
    arr, _vocab = load(fin, dtype=dtype, vocab=set(six.iterkeys(vocab)))
    # move from smaller index to avoid interferences between moves
    for k, ind in sorted(six.iteritems(vocab), key=lambda k_v: k_v[1]):
        _vocab, arr, _ = util.move_vocab(_vocab, arr, k, ind)

    return arr


def load(fin, dtype=np.float32, max_vocab=None, vocab=None):
    """
    Save word embedding file.
    Check :func:`word_embedding_loader.saver.glove.save` for the API.
    """
    npz = np.load(fin)
    arr = npz['arr']
    _vocab = dict(npz['vocab'])
    _vocab = {k: int(v) for k, v in six.iteritems(_vocab)}

    if vocab is not None:
        # aggregate items in list to allow editing _vocab
        for k in list(six.iterkeys(_vocab)):
            if k not in vocab:
                _vocab, arr, _ = util.remove_vocab(_vocab, arr, k)
    if max_vocab is not None:
        for k, v in list(six.iteritems(_vocab)):
            if v >= max_vocab:
                _vocab, arr, _ = util.remove_vocab(_vocab, arr, k)

    return arr.astype(dtype), _vocab
