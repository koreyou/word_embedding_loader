# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, \
    unicode_literals

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from word_embedding_loader import word_embedding
import word_embedding_loader.saver as saver


@pytest.fixture
def word_embedding_data():
    vocab = (
        (b'</s>', 0),
        (b'the', 1),
        ('日本語'.encode('utf-8'), 2)
    )
    vocab_dict = dict(vocab)
    arr = np.array(
        [[0.418, 0.24968, -0.41242, 0.1217],
         [0.013441, 0.23682, -0.16899, 0.40951],
         [0.15164, 0.30177, -0.16763, 0.17684]], dtype=np.float32)
    return arr, vocab, vocab_dict


@pytest.mark.parametrize("mod", [
    (saver.glove, 'glove', False),
    (saver.word2vec_bin, 'word2vec', True),
    (saver.word2vec_text, 'word2vec', False)
])
def test_save(word_embedding_data, mod, tmpdir):
    _saver, wtype, binary = mod
    arr_input, vocab_input, vocab_expected = word_embedding_data

    with open(tmpdir.join('output.txt').strpath, 'a+b') as f:
        _saver.save(f, arr_input, vocab_input)
        f.seek(0)
        obj = word_embedding.WordEmbedding.load(
            f.name, dtype=np.float32, format=wtype, binary=binary)
        vocab = obj.vocab
        arr = obj.vectors

    assert_array_equal(arr, arr_input)
    assert vocab_expected == vocab
