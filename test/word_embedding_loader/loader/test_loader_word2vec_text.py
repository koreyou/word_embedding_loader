# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, \
    unicode_literals

import io
import warnings

import numpy as np
import pytest
from numpy.testing import assert_array_equal

import word_embedding_loader.loader.word2vec_text as word2vec
from word_embedding_loader import ParseError, ParseWarning


def test_load(word2vec_text_file):
    arr, vocab = word2vec.load(word2vec_text_file)
    assert b'</s>' in vocab
    assert b'the' in vocab
    assert '日本語'.encode('utf-8') in vocab
    assert len(vocab) == 3
    assert len(arr) == 3
    assert arr.dtype == np.float32

    assert vocab[b'</s>'] == 0
    assert vocab[b'the'] == 1
    assert vocab['日本語'.encode('utf-8')] == 2

    assert_array_equal(arr[vocab[b'</s>']],
                       np.array([ 0.080054, 0.088388],
                                dtype=np.float32))
    assert_array_equal(arr[vocab[b'the']],
                       np.array([-1.420859, 1.156857],
                                dtype=np.float32))
    assert_array_equal(arr[vocab['日本語'.encode('utf-8')]],
                       np.array([-0.16799, 0.10951],
                                dtype=np.float32))


def test_check_valid():
    assert word2vec.check_valid(b"1 4",
                                b"the 0.418 0.24968 -0.41242 0.1217")
    assert not word2vec.check_valid(
        b"the 0.418 0.24968 -0.41242 0.1217",
        b", 0.013441 0.23682 -0.16899 0.40951")


def test_load_fail():
    f = io.BytesIO("""3 2
</s> 0.080054 0.088388
the -1.420859 1.156857
日本語 0.10951""".encode('utf-8'))
    with pytest.raises(ParseError):
        word2vec.load(f)


def test_load_warn():
    f = io.BytesIO("""3 2
</s> 0.080054 0.088388
the -1.420859 1.156857""".encode('utf-8'))

    with warnings.catch_warnings(record=True) as w:
        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")
        # Trigger a warning.
        arr, vocab = word2vec.load(f)
        # Verify some things
        assert len(w) == 1
        assert issubclass(w[-1].category, ParseWarning)
    assert len(vocab) == 2
    assert len(arr) == 2


def test_load_with_vocab(word2vec_text_file):
    vocab = dict((
        (b'</s>', 1),
        ('日本語'.encode('utf-8'), 0)
    ))

    arr = word2vec.load_with_vocab(word2vec_text_file, vocab)
    assert len(arr) == 2
    assert arr.dtype == np.float32
    # Machine epsilon is 5.96e-08 for float32
    assert_array_equal(arr[vocab[b'</s>']],
                       np.array([ 0.080054, 0.088388],
                                dtype=np.float32))
    assert_array_equal(arr[vocab['日本語'.encode('utf-8')]],
                       np.array([-0.16799, 0.10951],
                                dtype=np.float32))
