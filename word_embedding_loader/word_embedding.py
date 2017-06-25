# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

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
        raise OSError("Invalid format")


class LoadCondition(object):
    def __init__(self, mod, encoding, unicode_errors):
        self.mod = mod
        self.encoding = encoding
        self.unicode_errors = unicode_errors


class WordEmbedding(object):
    u"""
    Main API for loading and saving of pretrained word embedding files.

    .. note:: You do not need to call initializer directly in normal usage.
        Instead you should call
        :func:`~word_embedding_loader.word_embedding.WordEmbedding.load`.

    Args:
        vectors (numpy.ndarray): Word embedding representation vectors
        vocab (dict): Mapping from words (unicode) to vector
            indices (int).
        freqs (dict): Mapping from words (unicode) to word frequency counts
            (int).

    Attributes:
        vectors (numpy.ndarray): Word embedding vectors in shape of
            ``(vocabulary size, feature dimension)``.
        vocab (dict): Mapping from words (unicode) to vector indices (int).
        freqs (dict or None): Mapping from words (unicode) to frequency counts
            (int).

    """
    def __init__(self, vectors, vocab, freqs=None):
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
        self.freqs = freqs
        self._load_cond = None

    @classmethod
    def load(cls, path, vocab=None, dtype=np.float32, max_vocab=None,
             format=None, binary=False, encoding='utf-8',
             unicode_errors='strict'):
        u"""
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
            encoding (str): Encoding of the input file as defined in ``codecs``
                module of Python standard library.
            unicode_errors (str): Set the error handling scheme. The default
                error handler is 'strict' meaning that encoding errors raise
                ``ValueError``. Refer to ``codecs`` module for more information.

        Returns:
            :class:`~word_embedding_loader.word_embedding.WordEmbedding`
        """
        freqs = None
        if vocab is not None:
            with open(vocab, mode='r') as f:
                freqs = loader.vocab.load_vocab(
                    f, encoding=encoding, errors=unicode_errors)
            # Create vocab from freqs
            # [:None] gives all the list member
            vocab = {k: i for i, (k, v) in enumerate(
                    sorted(freqs.iteritems(),
                           key=lambda (k, v): v, reverse=True)[:max_vocab])}

        with open(path, mode='r') as f:
            if format is None:
                mod = classify_format(f)
            else:
                mod = _select_module(format, binary)
            if vocab is not None:
                arr = mod.loader.load_with_vocab(
                    f, vocab, dtype=dtype, encoding=encoding,
                    unicode_errors=unicode_errors)
                v = vocab
            else:
                arr, v = mod.loader.load(
                    f, max_vocab=max_vocab, dtype=dtype,
                    unicode_errors=unicode_errors, encoding=encoding)
        obj = cls(arr, v, freqs)
        obj._load_cond = LoadCondition(mod, encoding, unicode_errors)
        return obj

    def save(self, path, format, encoding='utf-8', unicode_errors='strict',
             binary=False, use_load_condition=False):
        u"""
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
                    "use_load_condition was specified but the object is not "
                    "loaded from a file")
            # Use load condition
            mod = self._load_cond.mod
            encoding = self._load_cond.encoding
            unicode_errors = self._load_cond.unicode_errors
        else:
            mod = _select_module(format, binary)

        def _mapper(item):
            key, value = item
            return (key.encode(encoding, errors=unicode_errors), value)

        if self.freqs is None:
            itr = map(_mapper, sorted(self.vocab.iteritems(), key=lambda (k, v): v))
        else:
            itr = map(_mapper, sorted(self.vocab.iteritems(), key=lambda (k, v): self.freqs[k], reverse=True))

        with open(path, mode='w') as f:
            mod.saver.save(f, self.vectors, itr)

    def __len__(self):
        return len(self.vectors)

    @property
    def size(self):
        u"""
        Feature dimension of the loaded vector.

        Returns:
            int
        """
        return self.vectors.shape[1]
