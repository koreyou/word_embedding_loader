# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

__all__ = ["WordEmbedding"]

import warnings

import numpy as np

from word_embedding_loader import loader, saver


# Mimick namespace
class _glove:
    loader = loader.glove
    saver = saver.glove


class _word2vec_bin:
    loader = loader.word2vec_bin
    saver = saver.word2vec_bin


class _word2vec_text:
    loader = loader.word2vec_text
    saver = saver.word2vec_text


def _select_module(format, binary):
    if format == 'glove':
        mod = _glove
        if binary:
            warnings.warn(
                "Argument binary=True for glove loader is ignored.",
                UserWarning)
    elif format == 'word2vec':
        if binary:
            mod = _word2vec_bin
        else:
            mod = _word2vec_text
    else:
        raise NameError('Unknown format "%s"' % format)
    return mod


def _get_two_lines(f):
    """
    Get the first and second lines
    Args:
        f (filelike):

    Returns:
        unicode

    """
    cur_pos = f.tell()
    l0 = f.readline()
    l1 = f.readline()
    f.seek(cur_pos)
    return l0, l1


def _classify_format(f):
    l0, l1 = _get_two_lines(f)
    if loader.glove.check_valid(l0, l1):
        return _glove
    elif loader.word2vec_text.check_valid(l0, l1):
        return _word2vec_text
    elif loader.word2vec_bin.check_valid(l0, l1):
        return _word2vec_bin
    else:
        raise OSError("Invalid format")


class LoadCondition(object):
    def __init__(self, mod, encoding, unicode_errors):
        self.mod = mod
        self.encoding = encoding
        self.unicode_errors = unicode_errors


class WordEmbedding(object):
    def __init__(self, vectors, vocab):
        if not isinstance(vectors, np.ndarray):
            raise TypeError(
                "Expected numpy.ndarray for vectors, %s found." % type(vectors))
        if not isinstance(vocab, dict):
            raise TypeError(
                "Expected dict for vocab, %s found." % type(vectors))
        if len(vectors) != len(vocab):
            warnings.warn(
                "vectors and vocab size unmatch (%d != %d)" %
                (len(vectors), len(vocab)))
        self.vectors = vectors
        self.vocab = vocab
        self._load_cond = None

    @classmethod
    def load(cls, path, dtype=np.float32, max_vocab=None, format=None,
             keep_order=False, encoding='utf-8', unicode_errors='strict',
             binary=False):
        u"""
        Load pretrained word embedding from a file.

        Args:
            path:
            dtype:
            max_vocab:
            format:
            keep_order:
            encoding:
            unicode_errors:
        """
        with open(path, mode='r') as f:
            if format is None:
                mod = _classify_format(f)
            else:
                mod = _select_module(format, binary)
            arr, vocab = mod.loader.load(
                f, max_vocab=max_vocab, dtype=dtype, keep_order=keep_order,
                unicode_errors=unicode_errors, encoding=encoding)
        obj = cls(arr, vocab)
        obj._load_cond = LoadCondition(mod, encoding, unicode_errors)
        return obj

    def save(self, path, format, encoding='utf-8', unicode_errors='strict',
             binary=False, use_load_condition=False):
        if use_load_condition:
            if self._load_cond is None:
                raise ValueError(
                    "use_load_condition was specified but the object is not loaded from a file")
            # Use load condition
            mod = self._load_cond.mod
            encoding = self._load_cond.encoding
            unicode_errors = self._load_cond.unicode_errors
        else:
            mod = _select_module(format, binary)

        with open(path, mode='w') as f:
            mod.saver.save(
                f, self.vectors, self.vocab, counts=None,
                encoding=encoding, unicode_errors=unicode_errors)

    def __len__(self):
        return len(self.vectors)

    @property
    def size(self):
        return self.vectors.shape[1]
