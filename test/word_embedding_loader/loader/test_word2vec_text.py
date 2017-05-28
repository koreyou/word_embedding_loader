# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import pytest
import numpy as np
from numpy.testing import assert_array_equal

import word_embedding_loader.loader.word2vec_text as word2vec


@pytest.mark.parametrize("keep_order", [True, False])
def test_load(word2vec_text_file, keep_order):
    arr, vocab = word2vec.load(word2vec_text_file, None,
                               dtype=np.float32, keep_order=keep_order)
    assert u'</s>' in vocab
    assert u'the' in vocab
    assert u'日本語' in vocab
    assert len(vocab) == 3
    assert len(arr) == 3
    assert arr.dtype == np.float32
    assert_array_equal(arr[vocab[u'</s>']],
                       np.array([ 0.080054, 0.088388],
                                dtype=np.float32))
    assert_array_equal(arr[vocab[u'the']],
                       np.array([-1.420859, 1.156857],
                                dtype=np.float32))
    assert_array_equal(arr[vocab[u'日本語']],
                       np.array([-0.16799, 0.10951],
                                dtype=np.float32))


def test_load_order(word2vec_text_file):
    arr, vocab = word2vec.load(
        word2vec_text_file, None, dtype=np.float32, keep_order=True)
    vocab_list = vocab.keys()
    assert vocab_list[0] == u'</s>'
    assert vocab_list[1] == u'the'
    assert vocab_list[2] == u'日本語'


def test_check_valid():
    assert word2vec.check_valid(u"1 4",
                                u"the 0.418 0.24968 -0.41242 0.1217")
    assert not word2vec.check_valid(
        u"the 0.418 0.24968 -0.41242 0.1217",
        u", 0.013441 0.23682 -0.16899 0.40951")
