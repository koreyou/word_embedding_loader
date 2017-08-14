# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, \
    unicode_literals
import six

__all__ = ["WordEmbedding", "classify_format"]

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
                b"Argument binary=True for glove loader is ignored.",
                UserWarning)
    elif format == 'word2vec':
        if binary:
            mod = _word2vec_bin
        else:
            mod = _word2vec_text
    else:
        raise NameError(('Unknown format "%s"' % format).encode('utf-8'))
    return mod


def _get_two_lines(f):
    """
    Get the first and second lines
    Args:
        f (filelike): File that is opened for ascii.

    Returns:
        bytes

    """
    cur_pos = f.tell()
    l0 = f.readline()
    l1 = f.readline()
    f.seek(cur_pos)
    return l0, l1


def classify_format(f):
    """
    Determine the format of word embedding file by their content. This operation
    only looks at the first two lines and does not check the sanity of input
    file.

    Args:
        f (Filelike):

    Returns:
        class

    """
    l0, l1 = _get_two_lines(f)
    if loader.glove.check_valid(l0, l1):
        return _glove
    elif loader.word2vec_text.check_valid(l0, l1):
        return _word2vec_text
    elif loader.word2vec_bin.check_valid(l0, l1):
        return _word2vec_bin
    else:
        raise OSError(b"Invalid format")


class WordEmbedding(object):
    """
    Main API for loading and saving of pretrained word embedding files.

    .. note:: You do not need to call initializer directly in normal usage.
        Instead you should call
        :func:`~word_embedding_loader.word_embedding.WordEmbedding.load`.

    Args:
        vectors (numpy.ndarray): Word embedding representation vectors
        vocab (dict): Mapping from words (bytes) to vector
            indices (int).
        freqs (dict): Mapping from words (bytes) to word frequency counts
            (int).

    Attributes:
        vectors (numpy.ndarray): Word embedding vectors in shape of
            ``(vocabulary size, feature dimension)``.
        vocab (dict): Mapping from words (bytes) to vector indices (int)
        freqs (dict or None): Mapping from words (bytes) to frequency counts
            (int).

    """
    def __init__(self, vectors, vocab, freqs=None):
        if not isinstance(vectors, np.ndarray):
            raise TypeError(
                ("Expected numpy.ndarray for vectors, %s found."% type(vectors)
                 ).encode('utf-8'))
        if not isinstance(vocab, dict):
            raise TypeError(
                ("Expected dict for vocab, %s found." % type(vectors)
                 ).encode('utf-8'))
        if len(vectors) != len(vocab):
            warnings.warn(
                ("vectors and vocab size unmatch (%d != %d)" %
                 (len(vectors), len(vocab))).encode('utf-8'))
        self.vectors = vectors
        self.vocab = vocab
        self.freqs = freqs
        self._load_cond = None

    @classmethod
    def load(cls, path, vocab=None, dtype=np.float32, max_vocab=None,
             format=None, binary=False):
        """
        Load pretrained word embedding from a file.

        Args:
            path (str): Path of file to load.
            vocab (str or None): Path to vocabulary file created by word2vec
                with ``-save-vocab <file>`` option. If vocab is given,
                :py:attr:`~vectors` and :py:attr:`~vocab` is ordered in
                descending order of frequency.
            dtype (numpy.dtype): Element data type to use for the array.
            max_vocab (int): Number of vocabulary to read.
            format (str or None): Format of the file. ``'word2vec'`` for file
                that was implemented in
                `word2vec <https://code.google.com/archive/p/word2vec/>`_,
                by Mikolov et al.. ``'glove'`` for file that was implemented in
                `GloVe <https://nlp.stanford.edu/projects/glove/>`_, Global
                Vectors for Word Representation, by Jeffrey Pennington,
                Richard Socher, Christopher D. Manning from Stanford NLP group.
                If ``None`` is given, the format is guessed from the content.
            binary (bool): Load file as binary file as in word embedding file
                created by
                `word2vec <https://code.google.com/archive/p/word2vec/>`_ with
                ``-binary 1`` option. If ``format`` is ``'glove'`` or ``None``,
                this argument is simply ignored

        Returns:
            :class:`~word_embedding_loader.word_embedding.WordEmbedding`
        """
        freqs = None
        if vocab is not None:
            with open(vocab, mode='rb') as f:
                freqs = loader.vocab.load_vocab(f)
            # Create vocab from freqs
            # [:None] gives all the list member
            vocab = {k: i for i, (k, v) in enumerate(
                     sorted(six.iteritems(freqs),
                            key=lambda k_v: k_v[1], reverse=True)[:max_vocab])}

        with open(path, mode='rb') as f:
            if format is None:
                mod = classify_format(f)
            else:
                mod = _select_module(format, binary)
            if vocab is not None:
                arr = mod.loader.load_with_vocab(f, vocab, dtype=dtype)
                v = vocab
            else:
                arr, v = mod.loader.load(f, max_vocab=max_vocab, dtype=dtype)

        obj = cls(arr, v, freqs)
        obj._load_cond = mod
        return obj

    def save(self, path, format, binary=False, use_load_condition=False):
        """
        Save object as word embedding file. For most arguments, you should refer
        to :func:`~word_embedding_loader.word_embedding.WordEmbedding.load`.

        Args:
            use_load_condition (bool): If `True`, options from
                :func:`~word_embedding_loader.word_embedding.WordEmbedding.load`
                is used.

        Raises:
            ValueError: ``use_load_condition == True`` but the object is not
                initialized via
                :func:`~word_embedding_loader.word_embedding.WordEmbedding.load`.
        """
        if use_load_condition:
            if self._load_cond is None:
                raise ValueError(
                    b"use_load_condition was specified but the object is not "
                    b"loaded from a file")
            # Use load condition
            mod = self._load_cond
        else:
            mod = _select_module(format, binary)


        if self.freqs is None:
            itr = list(
                sorted(six.iteritems(self.vocab), key=lambda k_v: k_v[1]))
        else:
            itr = list(
                sorted(six.iteritems(self.vocab),
                       key=lambda k_v: self.freqs[k_v[0]], reverse=True)
            )

        with open(path, mode='wb') as f:
            mod.saver.save(f, self.vectors, itr)

    def __len__(self):
        return len(self.vectors)

    @property
    def size(self):
        """
        Feature dimension of the loaded vector.

        Returns:
            int
        """
        return self.vectors.shape[1]
