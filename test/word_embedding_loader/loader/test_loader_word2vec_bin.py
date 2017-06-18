# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import numpy as np
import pytest
import word_embedding_loader.loader.word2vec_bin as word2vec
from numpy.testing import assert_allclose


def test_load(word2vec_bin_file):
    arr, vocab = word2vec.load(word2vec_bin_file)
    assert u'</s>' in vocab
    assert u'the' in vocab
    assert u'日本語' in vocab
    assert len(vocab) == 3
    assert len(arr) == 3
    assert arr.dtype == np.float32

    assert vocab[u'</s>'] == 0
    assert vocab[u'the'] == 1
    assert vocab[u'日本語'] == 2

    # Machine epsilon is 5.96e-08 for float32
    assert_allclose(arr[vocab[u'</s>']],
                    np.array([0.08005371, 0.08838806, -0.07660522,
                              -0.06556091, 0.02733154], dtype=np.float32),
                    atol=1e-8)
    assert_allclose(arr[vocab[u'日本語']],
                    np.array([-1.67798984, -0.02645044, -0.18966547,
                              1.16504729, -1.39292037], dtype=np.float32),
                    atol=1e-8)
    assert_allclose(arr[vocab[u'the']],
                    np.array([-0.5346821, 1.05223596, -0.24605329,
                              -1.82213438, -0.57173866], dtype=np.float32),
                    atol=1e-8)


def test_check_valid():
    assert word2vec.check_valid(u"2 4",
                                u"the 0.418 0.24968 -0.41242 0.1217")
    assert not word2vec.check_valid(
        u"the 0.418 0.24968 -0.41242 0.1217",
        u", 0.013441 0.23682 -0.16899 0.40951")


def test_load_with_vocab(word2vec_bin_file):
    vocab = dict((
        (u'</s>', 1),
        (u'日本語', 0)
    ))

    arr = word2vec.load_with_vocab(word2vec_bin_file, vocab)
    assert len(arr) == 2
    assert arr.dtype == np.float32
    # Machine epsilon is 5.96e-08 for float32
    assert_allclose(arr[vocab[u'</s>']],
                    np.array([0.08005371, 0.08838806, -0.07660522,
                              -0.06556091, 0.02733154], dtype=np.float32),
                    atol=1e-8)
    assert_allclose(arr[vocab[u'日本語']],
                    np.array([-1.67798984, -0.02645044, -0.18966547,
                              1.16504729, -1.39292037], dtype=np.float32),
                    atol=1e-8)
