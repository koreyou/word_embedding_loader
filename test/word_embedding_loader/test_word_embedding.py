# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, \
    unicode_literals

import io

import numpy as np
from numpy.testing import assert_array_equal
from six.moves import range

from word_embedding_loader import loader
from word_embedding_loader import word_embedding


def test__get_two_lines():
    f = io.BytesIO("""2 3
</s> 0.080054 0.088388 -0.07660
the -1.420859 1.156857 0.744776""".encode('utf-8'))
    l0, l1 = word_embedding._get_two_lines(f)
    # It s
    assert l0 == b"2 3\n"
    assert l1 == b"</s> 0.080054 0.088388 -0.07660\n"


class TestClassifyFormat:
    def test__classify_format_glove(self, glove_file):
        assert word_embedding.classify_format(glove_file) == word_embedding._glove

    def test__classify_format_word2vec_bin(self, word2vec_bin_file):
        assert word_embedding.classify_format(word2vec_bin_file) == word_embedding._word2vec_bin

    def test__classify_format_word2vec_text(self, word2vec_text_file):
        assert word_embedding.classify_format(word2vec_text_file) == word_embedding._word2vec_text


def test_WordEmbedding___init__():
    obj = word_embedding.WordEmbedding(
        np.zeros((123, 49), dtype=np.float32),
        dict.fromkeys(range(123), 0)
    )
    assert len(obj) == 123
    assert obj.size == 49


def test_WordEmbedding___load__(glove_file):
    """ Check one instance of loading; we assume that each loader is tested
    thoroughly in other unit test.
    """
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
