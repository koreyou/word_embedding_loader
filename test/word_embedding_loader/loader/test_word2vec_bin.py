from __future__ import absolute_import, print_function

import numpy as np
import pytest
import word_embedding_loader.loader.word2vec_bin as word2vec
from numpy.testing import assert_allclose


@pytest.mark.parametrize("keep_order", [True, False])
def test_load(word2vec_bin_file, keep_order):
    arr, vocab, scores = word2vec.load(word2vec_bin_file, 30, keep_order)
    assert u'</s>' in vocab
    assert u'the' in vocab
    assert u'of' in vocab
    assert len(vocab) == 3
    assert len(arr) == 3
    assert arr.dtype == np.float32
    # Machine epsilon is 5.96e-08 for float32
    assert_allclose(arr[vocab[u'</s>']],
                    np.array([0.08005371, 0.08838806, -0.07660522,
                              -0.06556091, 0.02733154], dtype=np.float32),
                    atol=1e-8)
    assert_allclose(arr[vocab[u'of']],
                    np.array([-1.67798984, -0.02645044, -0.18966547,
                              1.16504729, -1.39292037], dtype=np.float32),
                    atol=1e-8)
    assert_allclose(arr[vocab[u'the']],
                    np.array([-0.5346821, 1.05223596, -0.24605329,
                              -1.82213438, -0.57173866], dtype=np.float32),
                    atol=1e-8)

    assert scores is None


def test_load_order(word2vec_bin_file):
    arr, vocab, scores = word2vec.load(word2vec_bin_file, 30,
                                       keep_order=True)
    vocab_list = vocab.keys()
    assert vocab_list[0] == u'</s>'
    assert vocab_list[1] == u'the'
    assert vocab_list[2] == u'of'


def test_check_valid():
    assert word2vec.check_valid(u"2 4",
                                u"the 0.418 0.24968 -0.41242 0.1217")
    assert not word2vec.check_valid(
        u"the 0.418 0.24968 -0.41242 0.1217",
        u", 0.013441 0.23682 -0.16899 0.40951")
