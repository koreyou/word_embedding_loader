# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import StringIO

import pytest
import numpy as np
from numpy.testing import assert_array_equal

import word_embedding_loader.loader.glove as glove
from word_embedding_loader import ParseError


def test_load(glove_file):
    arr, vocab = glove.load(glove_file, dtype=np.float32)
    assert u'the' in vocab
    assert u',' in vocab
    assert u'日本語' in vocab
    assert len(vocab) == 3
    assert len(arr) == 3
    assert arr.dtype == np.float32

    assert vocab[u'the'] == 0
    assert vocab[u','] == 1
    assert vocab[u'日本語'] == 2

    assert_array_equal(arr[vocab[u'the']],
                       np.array([0.418, 0.24968, -0.41242, 0.1217],
                                dtype=np.float32))
    assert_array_equal(arr[vocab[u',']],
                       np.array([0.013441, 0.23682, -0.16899, 0.40951],
                                dtype=np.float32))
    assert_array_equal(arr[vocab[u'日本語']],
                       np.array([0.15164, 0.30177, -0.16763, 0.17684],
                                dtype=np.float32))


def test_check_valid():
    assert glove.check_valid(u"the 0.418 0.24968 -0.41242 0.1217",
                             u", 0.013441 0.23682 -0.16899 0.40951")
    assert not glove.check_valid(u"2 4", u"the 0.418 0.24968 -0.41242 0.1217")


def test_load_fail():
    f = StringIO.StringIO(u"""the 0.418 0.24968 -0.41242 0.1217
, 0.013441 0.23682 0.40951
日本語 0.15164 0.30177 -0.16763 0.17684""".encode('utf-8'))
    with pytest.raises(ParseError):
        glove.load(f)


def test_load_with_vocab(glove_file):
    vocab = dict((
        (u'the', 1),
        (u'日本語', 0)
    ))

    arr = glove.load_with_vocab(glove_file, vocab)
    assert len(arr) == 2
    assert arr.dtype == np.float32
    # Machine epsilon is 5.96e-08 for float32
    assert_array_equal(arr[vocab[u'the']],
                       np.array([0.418, 0.24968, -0.41242, 0.1217],
                                dtype=np.float32))
    assert_array_equal(arr[vocab[u'日本語']],
                       np.array([0.15164, 0.30177, -0.16763, 0.17684],
                                dtype=np.float32))
