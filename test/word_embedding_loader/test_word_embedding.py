# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, \
    unicode_literals

import io

import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
from six.moves import range
import pytest

from word_embedding_loader import loader
from word_embedding_loader import word_embedding


class TestClassifyFormat:
    def test__classify_format_glove(self, glove_file):
        assert word_embedding.classify_format(glove_file.name) == word_embedding._glove

    def test__classify_format_word2vec_bin(self, word2vec_bin_file):
        assert word_embedding.classify_format(word2vec_bin_file.name) == word_embedding._word2vec_bin

    def test__classify_format_word2vec_text(self, word2vec_text_file):
        assert word_embedding.classify_format(word2vec_text_file.name) == word_embedding._word2vec_text


def test_WordEmbedding___init__():
    obj = word_embedding.WordEmbedding(
        np.zeros((123, 49), dtype=np.float32),
        dict.fromkeys(range(123), 0)
    )
    assert len(obj) == 123
    assert obj.size == 49


def test_WordEmbedding___load__glove(glove_file):
    obj = word_embedding.WordEmbedding.load(glove_file.name)
    vocab = obj.vocab
    arr = obj.vectors
    assert b'the' in vocab
    assert b',' in vocab
    assert '日本語'.encode('utf-8') in vocab
    assert len(obj) == 3
    assert arr.dtype == np.float32

    assert obj._load_cond == word_embedding._glove

    assert_array_equal(arr[vocab[b'the']],
                       np.array([0.418, 0.24968, -0.41242, 0.1217],
                                dtype=np.float32))
    assert_array_equal(arr[vocab[b',']],
                       np.array([0.013441, 0.23682, -0.16899, 0.40951],
                                dtype=np.float32))
    assert_array_equal(arr[vocab['日本語'.encode('utf-8')]],
                       np.array([0.15164, 0.30177, -0.16763, 0.17684],
                                dtype=np.float32))


def test_WordEmbedding___load__word2vec_bin(word2vec_bin_file_path):
    """ I test word2vec too because it had problem when combined with
    other modules.
    """
    obj = word_embedding.WordEmbedding.load(word2vec_bin_file_path)
    vocab = obj.vocab
    arr = obj.vectors
    assert b'</s>' in vocab
    assert b'the' in vocab
    assert '日本語'.encode('utf-8') in vocab
    assert len(vocab) == 3
    assert len(arr) == 3
    assert arr.dtype == np.float32

    assert vocab[b'</s>'] == 0
    assert vocab[b'the'] == 1
    assert vocab['日本語'.encode('utf-8')] == 2

    # Machine epsilon is 5.96e-08 for float32
    assert_allclose(arr[vocab[b'</s>']],
                    np.array([0.08005371, 0.08838806, -0.07660522,
                              -0.06556091, 0.02733154], dtype=np.float32),
                    atol=1e-8)
    assert_allclose(arr[vocab['日本語'.encode('utf-8')]],
                    np.array([-1.67798984, -0.02645044, -0.18966547,
                              1.16504729, -1.39292037], dtype=np.float32),
                    atol=1e-8)
    assert_allclose(arr[vocab[b'the']],
                    np.array([-0.5346821, 1.05223596, -0.24605329,
                              -1.82213438, -0.57173866], dtype=np.float32),
                    atol=1e-8)


def test_WordEmbedding___load__typeerror(glove_file):
    with pytest.raises(TypeError):
        word_embedding.WordEmbedding.load(glove_file.name, vocab={})


def test_WordEmbedding___load__vocab_set(glove_file):
    vocab = {b',', '日本語'.encode('utf-8')}
    obj = word_embedding.WordEmbedding.load(glove_file.name, vocab=vocab)
    vocab = obj.vocab
    arr = obj.vectors
    assert b',' in vocab
    assert '日本語'.encode('utf-8') in vocab
    assert len(obj) == 2
    assert arr.dtype == np.float32

    assert obj._load_cond == word_embedding._glove

    assert_array_equal(arr[vocab[b',']],
                       np.array([0.013441, 0.23682, -0.16899, 0.40951],
                                dtype=np.float32))
    assert_array_equal(arr[vocab['日本語'.encode('utf-8')]],
                       np.array([0.15164, 0.30177, -0.16763, 0.17684],
                                dtype=np.float32))


def test_WordEmbedding___load___vocab(word2vec_text_file, vocab_file):
    obj = word_embedding.WordEmbedding.load(
        word2vec_text_file.name, vocab=vocab_file.name)
    vocab = obj.vocab
    arr = obj.vectors
    assert b'</s>' in vocab
    assert b'the' not in vocab
    assert '日本語'.encode('utf-8') in vocab
    assert len(obj) == 2
    assert arr.dtype == np.float32

    assert vocab['日本語'.encode('utf-8')] == 0
    assert vocab[b'</s>'] == 1

    assert_array_equal(arr[vocab[b'</s>']],
                       np.array([ 0.080054, 0.088388],
                                dtype=np.float32))
    assert_array_equal(arr[vocab['日本語'.encode('utf-8')]],
                       np.array([-0.16799, 0.10951],
                                dtype=np.float32))


def test_WordEmbedding___load___vocab_maxvocab(word2vec_text_file, vocab_file):
    obj = word_embedding.WordEmbedding.load(
        word2vec_text_file.name, vocab=vocab_file.name, max_vocab=1)
    vocab = obj.vocab
    arr = obj.vectors
    assert b'</s>' not in vocab
    assert b'the' not in vocab
    assert '日本語'.encode('utf-8') in vocab
    assert len(obj) == 1
    assert arr.dtype == np.float32

    assert vocab['日本語'.encode('utf-8')] == 0
    assert_array_equal(arr[vocab['日本語'.encode('utf-8')]],
                       np.array([-0.16799, 0.10951],
                                dtype=np.float32))


def test_WordEmbedding___save__(tmpdir):
    vocab = {
        b'</s>': 0,
        b'the': 1,
        '日本語'.encode('utf-8'): 2
    }
    arr_input = np.array(
        [[0.418, 0.24968, -0.41242, 0.1217],
         [0.013441, 0.23682, -0.16899, 0.40951],
         [0.15164, 0.30177, -0.16763, 0.17684]], dtype=np.float32)

    obj = word_embedding.WordEmbedding(arr_input, vocab)
    tmp_path = tmpdir.join('WordEmbedding__save.txt').strpath
    obj.save(tmp_path, format="word2vec", binary=True)
    with open(tmp_path, 'rb') as f:
        arr, vocab = loader.word2vec_bin.load(f, dtype=np.float32)
    assert_array_equal(arr, arr_input)
    assert vocab == vocab


def test_WordEmbedding_resize():
    vocab = {
        b'</s>': 0,
        b'the': 1,
        '日本語'.encode('utf-8'): 2
    }
    arr_input = np.array(
        [[0.418, 0.24968, -0.41242, 0.1217],
         [0.013441, 0.23682, -0.16899, 0.40951],
         [0.15164, 0.30177, -0.16763, 0.17684]], dtype=np.float32)

    obj = word_embedding.WordEmbedding(arr_input, vocab)
    obj.resize(2)

    vocab_e = {
        b'</s>': 0,
        b'the': 1
    }
    arr_e = np.array(
        [[0.418, 0.24968, -0.41242, 0.1217],
         [0.013441, 0.23682, -0.16899, 0.40951]], dtype=np.float32)

    assert_array_equal(obj.vectors, arr_e)
    assert obj.vocab == vocab_e


def test_WordEmbedding_resize_freq():
    vocab = {
        b'</s>': 0,
        b'the': 1,
        '日本語'.encode('utf-8'): 2
    }
    freqs = {
        b'</s>': 100,
        b'the': 1,
        '日本語'.encode('utf-8'): 2
    }
    arr_input = np.array(
        [[0.418, 0.24968, -0.41242, 0.1217],
         [0.013441, 0.23682, -0.16899, 0.40951],
         [0.15164, 0.30177, -0.16763, 0.17684]], dtype=np.float32)

    obj = word_embedding.WordEmbedding(arr_input, vocab, freqs=freqs)
    obj.resize(2)

    vocab_e = {
        b'</s>': 0,
        '日本語'.encode('utf-8'): 1
    }
    arr_e = np.array(
        [[0.418, 0.24968, -0.41242, 0.1217],
         [0.15164, 0.30177, -0.16763, 0.17684]], dtype=np.float32)
    freqs_e = {
        b'</s>': 100,
        '日本語'.encode('utf-8'): 2
    }
    assert_array_equal(obj.vectors, arr_e)
    assert obj.vocab == vocab_e
    assert obj.freqs == freqs_e


def test_load_word_embedding_random(word2vec_text_file, vocab_file):
    arr, vocab = word_embedding.load_word_embedding(
        word2vec_text_file.name, vocab=vocab_file.name, unk=b'<unk>',
        unk_index=0, eos=b'</s>', eos_index=1, unk_creation_method='random')

    vocab_e = {
        b'<unk>': 0,
        b'</s>': 1,
        '日本語'.encode('utf-8'): 2
    }
    assert vocab == vocab_e
    assert len(arr) == 3
    assert arr.dtype == np.float32
    assert_array_equal(
        arr[1], np.array([0.080054, 0.088388], dtype=np.float32))
    assert_array_equal(
        arr[2], np.array([-0.16799, 0.10951], dtype=np.float32))
    arr_unk = arr[0]

    # make sure that arr[0] is preserved amongst different runs.
    arr, _ = word_embedding.load_word_embedding(
        word2vec_text_file.name, vocab=vocab_file.name, unk=b'<unk>',
        unk_index=0, eos=b'</s>', eos_index=1, unk_creation_method='random')
    assert_array_equal(arr[0], arr_unk)


def test_load_word_embedding_least_common(glove_file):
    arr, vocab = word_embedding.load_word_embedding(
        glove_file.name, vocab=None,  max_vocab=4, unk=b'<unk>', unk_index=1,
        eos=b'</s>', eos_index=0, unk_creation_method='least_common',
        unk_creation_n_least=2
    )

    vocab_e = {
        b'</s>': 0,
        b'<unk>': 1,
        b'the': 2,
        b',': 3
    }
    assert vocab == vocab_e
    assert len(arr) == 4
    assert arr.dtype == np.float32
    assert_array_equal(
        arr[2], np.array([0.418, 0.24968, -0.41242, 0.1217], dtype=np.float32))
    assert_array_equal(
        arr[3], np.array([0.013441, 0.23682, -0.16899, 0.40951], dtype=np.float32))
    # unk is produced by averaging "," and "日本語"
    assert_array_equal(
        arr[1], np.array([0.0825405, 0.26929501, -0.16831, 0.29317498], dtype=np.float32))


def test_load_word_embedding_no_unk(word2vec_bin_file_path):
    arr, vocab = word_embedding.load_word_embedding(
        word2vec_bin_file_path, vocab=None, unk=None, unk_index=1, eos=b'</s>',
        eos_index=1, max_vocab=3)

    vocab_e = {
        b'the': 0,
        b'</s>': 1,
        '日本語'.encode('utf-8'): 2
    }
    assert vocab == vocab_e
    assert len(arr) == 3
    assert arr.dtype == np.float32
    # Machine epsilon is 5.96e-08 for float32
    a = np.array(
        [-0.5346821, 1.05223596, -0.24605329, -1.82213438, -0.57173866],
        dtype=np.float32)
    assert_allclose(arr[0], a, atol=1e-8)
    a = np.array(
        [0.08005371, 0.08838806, -0.07660522, -0.06556091, 0.02733154],
        dtype=np.float32)
    assert_allclose(arr[1], a, atol=1e-8)
    a = np.array(
        [-1.67798984, -0.02645044, -0.18966547, 1.16504729, -1.39292037],
         dtype=np.float32)
    assert_allclose(arr[2], a, atol=1e-8)
