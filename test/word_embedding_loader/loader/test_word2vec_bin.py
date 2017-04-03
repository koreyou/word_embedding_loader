from __future__ import absolute_import, print_function
import os

import pytest
import numpy as np
from numpy.testing import assert_allclose

import word_embedding_loader.loader.word2vec_bin as word2vec


@pytest.fixture
def load_word2vec_bin():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(cur_dir, 'word2vec.bin')
    ret = word2vec.load(path, 30)
    return ret


def test_load(load_word2vec_bin):
    arr, vocab, scores, ranks = load_word2vec_bin
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
    assert ranks[vocab[u'</s>']] == 0
    assert ranks[vocab[u'the']] == 1
    assert ranks[vocab[u'of']] == 2