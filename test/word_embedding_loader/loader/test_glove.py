# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import StringIO

import pytest
import numpy as np
from numpy.testing import assert_array_equal

import word_embedding_loader.loader.glove as glove


def test_load(glove_file):
    arr, vocab, scores, ranks = glove.load(glove_file, None, dtype=np.float32)
    assert u'the' in vocab
    assert u',' in vocab
    assert u'.' in vocab
    assert len(vocab) == 3
    assert len(arr) == 3
    assert arr.dtype == np.float32
    assert_array_equal(arr[vocab[u'the']],
                       np.array([0.418, 0.24968, -0.41242, 0.1217],
                                dtype=np.float32))
    assert_array_equal(arr[vocab[u',']],
                       np.array([0.013441, 0.23682, -0.16899, 0.40951],
                                dtype=np.float32))
    assert_array_equal(arr[vocab[u'.']],
                       np.array([0.15164, 0.30177, -0.16763, 0.17684],
                                dtype=np.float32))
    assert scores is None
    assert ranks[vocab[u'the']] == 0
    assert ranks[vocab[u',']] == 1
    assert ranks[vocab[u'.']] == 2


def test_check_valid():
    assert glove.check_valid(u"the 0.418 0.24968 -0.41242 0.1217",
                             u", 0.013441 0.23682 -0.16899 0.40951")
    assert not glove.check_valid(u"2 4", u"the 0.418 0.24968 -0.41242 0.1217")
