# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, \
    unicode_literals

from word_embedding_loader import WordEmbedding
from word_embedding_loader import util
import numpy as np
from numpy.testing import assert_allclose
from numpy.testing import assert_array_equal


def test_infer_unk_id():
    vocab = {
        b'the': 0,
        b',': 1,
        "日本語".encode('utf-8'):2,
        b'unk': 3,
        b'<unk>': 4
    }
    r = util.infer_unk_id(vocab)
    assert r == b'<unk>'


def test_create_unk_least_common():
    vectors = np.array(
      [[ 0.86148602,  0.65806735],
       [ 0.44499034,  0.20795698],
       [ 0.17485058,  0.02284675],
       [ 0.12795962,  0.56146187],
       [ 0.2376053 ,  0.33676073]], dtype=np.float32)
    vocab = {
        b'the': 0,
        b',': 1,
        "日本語".encode('utf-8'):2,
        b'.': 3,
        b'<unk>': 4
    }
    we = WordEmbedding(vectors, vocab)
    ret = util.create_unk_least_common(we, 3)
    expected = np.array([ 0.18013851,  0.30702314], dtype=np.float32)
    assert_allclose(ret, expected)
    freqs = {
        b'the': 500,
        b',': 0,
        "日本語".encode('utf-8'):300,
        b'.': 400,
        b'<unk>': 0
    }
    we = WordEmbedding(vectors, vocab, freqs)
    ret = util.create_unk_least_common(we, 3)
    expected = np.array([0.28581539, 0.18918817], dtype=np.float32)
    assert_allclose(ret, expected)


def test_move_vocab():
    vectors = np.array(
      [[ 0.86148602,  0.65806735],
       [ 0.44499034,  0.20795698],
       [ 0.17485058,  0.02284675],
       [ 0.12795962,  0.56146187],
       [ 0.2376053 ,  0.33676073]],dtype=np.float32)

    # Move back
    vocab = {
        b'the': 0,
        b',': 1,
        "日本語".encode('utf-8'):2,
        b'.': 3,
        b'<unk>': 4
    }
    vocab_r, vectors_r, old_index = util.move_vocab(vocab, vectors, b',', 4)
    vectors_e = np.array(
      [[ 0.86148602,  0.65806735],
       [ 0.17485058,  0.02284675],
       [ 0.12795962,  0.56146187],
       [ 0.2376053 ,  0.33676073],
       [ 0.44499034,  0.20795698]],dtype=np.float32)
    vocab_e = {
        b'the': 0,
        "日本語".encode('utf-8'):1,
        b'.': 2,
        b'<unk>': 3,
        b',': 4,
    }
    assert vocab_r == vocab_e
    assert_array_equal(vectors_r, vectors_e)
    assert old_index == 1

    # Move foward
    vocab = {
        b'the': 0,
        b',': 1,
        "日本語".encode('utf-8'):2,
        b'.': 3,
        b'<unk>': 4
    }
    vocab_r, vectors_r, old_index = util.move_vocab(
        vocab, vectors, "日本語".encode('utf-8'), 1)
    vectors_e = np.array(
      [[ 0.86148602,  0.65806735],
       [ 0.17485058,  0.02284675],
       [ 0.44499034,  0.20795698],
       [ 0.12795962,  0.56146187],
       [ 0.2376053 ,  0.33676073]],dtype=np.float32)
    vocab_e = {
        b'the': 0,
        "日本語".encode('utf-8'): 1,
        b',': 2,
        b'.': 3,
        b'<unk>': 4
    }
    assert vocab_r == vocab_e
    assert_array_equal(vectors_r, vectors_e)
    assert old_index == 2

    # No moving
    vocab = {
        b'the': 0,
        b',': 1,
        "日本語".encode('utf-8'):2,
        b'.': 3,
        b'<unk>': 4
    }
    vocab_r, vectors_r, old_index = util.move_vocab(vocab, vectors, b'<unk>', 4)
    assert vocab_r == vocab
    assert_array_equal(vectors_r, vectors)
    assert old_index == 4


def test_insert_vocab():
    vectors = np.array(
      [[ 0.86148602,  0.65806735],
       [ 0.44499034,  0.20795698],
       [ 0.17485058,  0.02284675],
       [ 0.12795962,  0.56146187]],dtype=np.float32)
    vec = np.array([ 0.2376053 ,  0.33676073], dtype=np.float32)

    vocab = {
        b'the': 0,
        b',': 1,
        "日本語".encode('utf-8'):2,
        b'.': 3
    }
    vocab_r, vectors_r = util.insert_vocab(vocab, vectors, b'<unk>', 2, vec)
    vectors_e = np.array(
      [[ 0.86148602,  0.65806735],
       [ 0.44499034,  0.20795698],
       [ 0.2376053,   0.33676073],
       [ 0.17485058,  0.02284675],
       [ 0.12795962,  0.56146187]],dtype=np.float32)
    vocab_e = {
        b'the': 0,
        b',': 1,
        b'<unk>': 2,
        "日本語".encode('utf-8'): 3,
        b'.': 4
    }
    assert vocab_r == vocab_e
    assert_array_equal(vectors_r, vectors_e)

    # Insert to the end
    vocab = {
        b'the': 0,
        b',': 1,
        "日本語".encode('utf-8'):2,
        b'.': 3
    }
    vocab_r, vectors_r = util.insert_vocab(vocab, vectors, b'<unk>', 4, vec)
    vectors_e = np.array(
        [[0.86148602, 0.65806735],
         [0.44499034, 0.20795698],
         [0.17485058, 0.02284675],
         [0.12795962, 0.56146187],
         [0.2376053,  0.33676073]], dtype=np.float32)
    vocab_e = {
        b'the': 0,
        b',': 1,
        "日本語".encode('utf-8'): 2,
        b'.': 3,
        b'<unk>': 4
    }
    assert vocab_r == vocab_e
    assert_array_equal(vectors_r, vectors_e)


def test_remove_vocab():
    vectors = np.array(
      [[ 0.86148602,  0.65806735],
       [ 0.44499034,  0.20795698],
       [ 0.17485058,  0.02284675],
       [ 0.12795962,  0.56146187],
       [ 0.2376053 ,  0.33676073]],dtype=np.float32)

    vocab = {
        b'the': 0,
        b',': 1,
        "日本語".encode('utf-8'):2,
        b'.': 3,
        b'<unk>': 4
    }
    vocab_r, vectors_r, old_index = util.remove_vocab(vocab, vectors, b',')
    vectors_e = np.array(
      [[ 0.86148602,  0.65806735],
       [ 0.17485058,  0.02284675],
       [ 0.12795962,  0.56146187],
       [ 0.2376053 ,  0.33676073]],dtype=np.float32)
    vocab_e = {
        b'the': 0,
        "日本語".encode('utf-8'): 1,
        b'.': 2,
        b'<unk>': 3
    }
    assert vocab_r == vocab_e
    assert_array_equal(vectors_r, vectors_e)
    assert old_index == 1

    # Removed from very back
    vocab = {
        b'the': 0,
        b',': 1,
        "日本語".encode('utf-8'):2,
        b'.': 3,
        b'<unk>': 4
    }
    vocab_r, vectors_r, old_index = util.remove_vocab(vocab, vectors, b'<unk>')
    vectors_e = np.array(
      [[ 0.86148602,  0.65806735],
       [ 0.44499034,  0.20795698],
       [ 0.17485058,  0.02284675],
       [ 0.12795962,  0.56146187]],dtype=np.float32)
    vocab_e = {
        b'the': 0,
        b',': 1,
        "日本語".encode('utf-8'): 2,
        b'.': 3
    }

    assert vocab_r == vocab_e
    assert_array_equal(vectors_r, vectors_e)
    assert old_index == 4
