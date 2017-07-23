# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, \
    unicode_literals

import numpy as np
import pytest
from numpy.testing import assert_array_equal

import word_embedding_loader.loader as loader
import word_embedding_loader.saver as saver


@pytest.fixture
def word_embedding_data():
    vocab = (
        (b'</s>', 0),
        (b'the', 1),
        ('日本語'.encode('utf-8'), 2)
    )
    vocab_dict = {
        u'</s>': 0,
        u'the': 1,
        u'日本語': 2
    }
    arr = np.array(
        [[0.418, 0.24968, -0.41242, 0.1217],
         [0.013441, 0.23682, -0.16899, 0.40951],
         [0.15164, 0.30177, -0.16763, 0.17684]], dtype=np.float32)
    return arr, vocab, vocab_dict


@pytest.mark.parametrize("mod", [
    (saver.glove, loader.glove),
    (saver.word2vec_bin, loader.word2vec_bin),
    (saver.word2vec_text, loader.word2vec_text)
])
def test_load(word_embedding_data, mod, tmpdir):
    _saver, _loader = mod
    arr_input, vocab_input, vocab_expected = word_embedding_data

    with open(tmpdir.join('output.txt').strpath, 'a+b') as f:
        _saver.save(f, arr_input, vocab_input)
        f.seek(0)
        arr, vocab = _loader.load(f, dtype=np.float32, encoding='utf-8')
    assert_array_equal(arr, arr_input)
    assert vocab_expected == vocab
