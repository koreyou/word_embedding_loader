# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import StringIO

import pytest
import numpy as np
from numpy.testing import assert_array_equal

import word_embedding_loader.loader.word2vec_text as word2vec


@pytest.fixture
def load_word2vec_text():
    f = StringIO.StringIO(u"""2 3
</s> 0.080054 0.088388 -0.07660
the -1.420859 1.156857 0.744776""")
    return word2vec.load(f, None, dtype=np.float32)


def test_load(load_word2vec_text):
    arr, vocab, scores, ranks = load_word2vec_text
    assert u'</s>' in vocab
    assert u'the' in vocab
    assert len(vocab) == 2
    assert len(arr) == 2
    assert arr.dtype == np.float32
    assert_array_equal(arr[vocab[u'</s>']],
                       np.array([ 0.080054, 0.088388, -0.07660],
                                dtype=np.float32))
    assert_array_equal(arr[vocab[u'the']],
                       np.array([-1.420859, 1.156857, 0.744776],
                                dtype=np.float32))

    assert scores is None
    assert ranks[vocab[u'</s>']] == 0
    assert ranks[vocab[u'the']] == 1


def test_check_valid():
    assert word2vec.check_valid(u"1 4",
                                u"the 0.418 0.24968 -0.41242 0.1217")
    assert not word2vec.check_valid(
        u"the 0.418 0.24968 -0.41242 0.1217",
        u", 0.013441 0.23682 -0.16899 0.40951")
