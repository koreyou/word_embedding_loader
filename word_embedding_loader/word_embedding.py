# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, \
    unicode_literals
import six

__all__ = ["WordEmbedding", "classify_format"]

import warnings

import numpy as np

from word_embedding_loader import loader, saver, util


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


class _numpy:
    loader = loader.numpy
    saver = saver.numpy


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
    elif format == 'numpy':
        mod = _numpy
        if binary:
            warnings.warn(
                b"Argument binary=True for numpy loader is ignored.",
                UserWarning)
    else:
        raise NameError(('Unknown format "%s"' % format).encode('utf-8'))
    return mod


def classify_format(path):
    """
    Determine the format of word embedding file by their content. This operation
    only looks at the first two lines and does not check the sanity of input
    file.

    Args:
        path (str):

    Returns:
        class

    """
    if loader.glove.check_valid(path):
        return _glove
    elif loader.word2vec_text.check_valid(path):
        return _word2vec_text
    elif loader.numpy.check_valid(path):
        return _numpy
    elif loader.word2vec_bin.check_valid(path):
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
            vocab (str or set or None): If ``str``, it is assumed to be path to
                vocabulary file created by word2vec with
                ``-save-vocab <file>`` option.
                If ``set`` it will only load words that are in vocab.
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
        vocab_dict = None
        if isinstance(vocab, six.string_types):
            with open(vocab, mode='rb') as f:
                freqs = loader.vocab.load_vocab(f)
            # Create vocab from freqs
            # [:None] gives all the list member
            vocab_dict = {k: i for i, (k, v) in enumerate(
                 sorted(six.iteritems(freqs),
                 key=lambda k_v: k_v[1], reverse=True)[:max_vocab])}
        elif isinstance(vocab, list):
            vocab = set(vocab)
        elif isinstance(vocab, set) or vocab is None:
            pass
        else:
            raise TypeError(
                'Expected set, str or None for vocab but %s is given.' %
                type(vocab)
            )

        if format is None:
            mod = classify_format(path)
        else:
            mod = _select_module(format, binary)
        with open(path, mode='rb') as f:
            if vocab_dict is not None:
                arr = mod.loader.load_with_vocab(f, vocab_dict, dtype=dtype)
            else:
                arr, vocab_dict = mod.loader.load(
                    f, max_vocab=max_vocab, dtype=dtype, vocab=vocab)

        obj = cls(arr, vocab_dict, freqs)
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

    def resize(self, size):
        """
        Reduce number of vocabulary in place.

        Args:
            size (int): new size

        Returns:
            ~WordEmbedding: Returns reference to self
        """
        if size < len(self):
            n = len(self) - size
            if self.freqs is not None:
                del_keys = [
                    k for k, v in sorted(
                        six.iteritems(self.freqs), key=lambda k_v: k_v[1])[:n]]
            else:
                del_keys = [
                    k for k, v in six.iteritems(self.vocab) if v >= size]
            assert len(del_keys) == n
            for k in del_keys:
                self.vocab, self.vectors, _ = \
                    util.remove_vocab(self.vocab, self.vectors, k)
                if self.freqs is not None:
                    del self.freqs[k]
        return self

    def apply_freqs(self, freqs=None):
        if freqs is None:
            if self.freqs is None:
                raise ValueError("You must supply freqs when self.freqs is None")
            return self.apply_freqs(self.freqs)
        if len(freqs) != len(self.vocab):
            raise ValueError("len(freqs) != len(self.vocab)")
        self.freqs = freqs
        new_vocab = {}
        # sort then reverse for stability
        freqs_list = sorted(six.iteritems(freqs), key=lambda k_v: k_v[1])
        for i, (k, _) in enumerate(freqs_list[::-1]):
            new_vocab[k] = i


def load_word_embedding(
        path, vocab=None, dtype=np.float32, max_vocab=None,
        format=None, binary=False, unk=b'<unk>', unk_index=0, eos=b'</s>',
        eos_index=1, unk_creation_method='least_common',
        unk_creation_n_least=10, random_init_std='auto', random_init_seed=0):
    """
    Load pretrained word embedding from a file. This is short hand method for
    using :func:`~word_embedding_loader.word_embedding.WordEmbedding.load` and
    getting :attr:`~word_embedding_loader.word_embedding.WordEmbedding.vectors`
    and :attr:`~word_embedding_loader.word_embedding.WordEmbedding.vocab`.
    This is especially useful if you have no plans to save the word embedding.

    By specifying ``unk`` or ``eos``, you are guaranteed to have a embedding
    for out-of-vocabulary words and for the end-of-sentence token, respectively.
    The function first tries to detect if those tokens already exists by
    looking for words that are they tend to be associated with (e.g. "</s>").
    On success, it replaces those found vocabularies with the words you have
    provided to the function. On failure, it initializes and append new
    embeddings to the word embedding.

    Args:
        path (str): Path of file to load.
        vocab (str or set or None): Path to vocab files or set of vocab to use.
            Refer
            :func:`~word_embedding_loader.word_embedding.WordEmbedding.load`
            for details.
        dtype (numpy.dtype): Element data type to use for the array.
        max_vocab (int): Number of vocabulary to read.
        format (str or None): Format of the file. Refer
            :func:`~word_embedding_loader.word_embedding.WordEmbedding.load` for
            details.
        binary (bool): Load file as binary file.  Refer
            :func:`~word_embedding_loader.word_embedding.WordEmbedding.load` for
            details.
        unk (bytes or None): The vocabulary for out-of-vocabulary words.
            If ``None`` it will not do any post-precessings to gurentee that
            it exists.
        unk_index (int): Index to which unk is inserted.
        eos (bytes or None): The vocabulary for the end-of-sentence token.
            If ``None`` it will not do any post-precessings to gurentee that
            it exists.
        eos_index (int): Index to which eos is inserted.
        unk_creation_method (str): The method to use if it needs to create
            an embedding for out-of-vocabulary words. ``'least_common'`` will
            take n least common word embeddings and take their average (it will
            use frequencies from vocab file if provided or assume that words
            are ordered in the order of frequencies). ``random`` will create
            word embedding normal distribution drawn with the
            ``random_init_std`` argument. This argument is ignored if ``unk``
            token is already available in loaded word embedding.
        unk_creation_n_least (str): number of least common word embeddings to
            use when ``unk_creation_method`` is ``'least_common'``.
        random_init_std (float or 'auto'): standard deviation of newly created
            ``unk`` and ``eos`` tokens (only when they are to be randomly
            initialized). Specify ``'auto'`` to use standard deviation of
            the loaded word embeddings.
        random_init_seed (int or None): Random seed that is used to create
            unk or eos word embedding. Default is ``0`` so you get the same
            vector every time you use the function (this is useful if you want
            to use trained model). You should specify `None` to do experiments
            with different seed, but you should explicitly save the word
            embedding to use it with the trained model.

    Returns:
        vectors (numpy.ndarray): Word embedding vectors in shape of
            ``(vocabulary size, feature dimension)``.
        vocab (dict): Mapping from words (bytes) to vector indices (int)
    """
    if max_vocab is not None:
        if eos_index >= max_vocab:
            raise ValueError('eos_index must be within max_vocab')
        if unk_index >= max_vocab:
            raise ValueError('unk_index must be within max_vocab')
    if unk is not None and eos is not None and eos_index == unk_index:
        raise ValueError('eos_index and unk_index must be different')
    EOS_CAND = b'</s>'
    if unk is None and eos is None:
        _vocab = vocab
        _max_vocab = max_vocab
    elif unk is not None and unk_creation_method == 'least_common':
        # Just load everything
        _vocab = None
        _max_vocab = None
    else:
        if isinstance(vocab, six.string_types):
            # it is difficult to add words to vocab so apply frequency afterward
            _vocab = None
            _max_vocab = None
        elif isinstance(vocab, (list, set)):
            _vocab = set(vocab)
            _max_vocab = max_vocab
            if unk is not None:
                for w in util.UNK_ID_CANDIDATES:
                    _vocab.add(w)
            if eos is not None:
                _vocab.add(EOS_CAND)
        else:
            _max_vocab = None
            _vocab = None
    we = WordEmbedding.load(path, vocab=_vocab, dtype=dtype,
                            max_vocab=_max_vocab, format=format, binary=binary)
    if unk is None and eos is None:
        return we.vectors, we.vocab

    if eos_index >= len(we):
        raise ValueError('eos_index must be smaller than the word embedding size')
    if unk_index >= len(we):
        raise ValueError('unk_index must be smaller than the word embedding size')

    if isinstance(vocab, six.string_types):
        # vocab path was ignored
        with open(vocab, mode='rb') as f:
            freqs_loaded = loader.vocab.load_vocab(f)
        # balance vocab and freqs
        freqs = {k: freqs_loaded.get(k, 0) for k in six.iterkeys(we.vocab)}
        max_freq = max(freqs.values())
        we.apply_freqs(freqs)

    if we.freqs is not None:
        max_freq = max(we.freqs.values())

    if unk is not None:
        if eos is not None and unk_index > eos_index:
            # Inserting eos will move index of unk so compensate for that
            unk_index -= 1
        _unk = util.infer_unk_id(we.vocab)
        if _unk is None:
            # Create unk
            if unk_creation_method == 'least_common':
                v = util.create_unk_least_common(we, unk_creation_n_least)
            elif unk_creation_method == 'random':
                if random_init_std == 'auto':
                    random_init_std = np.std(we.vectors)
                v = util.create_random_vector(
                    we.vectors.shape[1], random_init_std,
                    dtype=we.vectors.dtype, seed=random_init_seed)
            else:
                raise ValueError(
                    'Unknown unk_creation_method %s' % unk_creation_method)
            we.vocab, we.vectors = util.insert_vocab(
                we.vocab, we.vectors, unk, unk_index, v)
        else:
            we.vocab, we.vectors, _ = util.move_vocab(
                we.vocab, we.vectors, _unk, unk_index)
            del we.vocab[_unk]
            we.vocab[unk] = unk_index
        if we.freqs is not None:
            we.freqs[unk] = max_freq + 1
    if eos is not None:
        if EOS_CAND not in we.vocab:
            # Create eos
            if random_init_std == 'auto':
                random_init_std = np.std(we.vectors)
            random_init_seed_eos = (None if random_init_seed is None else
                                    random_init_seed + 1)
            v = util.create_random_vector(
                we.vectors.shape[1], random_init_std, dtype=we.vectors.dtype,
                seed=random_init_seed_eos
            )
            we.vocab, we.vectors = util.insert_vocab(
                we.vocab, we.vectors, eos, eos_index, v)
        else:
            we.vocab, we.vectors, _ = util.move_vocab(
                we.vocab, we.vectors, EOS_CAND, eos_index)
            del we.vocab[EOS_CAND]
            we.vocab[eos] = eos_index
        if we.freqs is not None:
            we.freqs[eos] = max_freq + 1

    if isinstance(vocab, six.string_types):
        # Prune freq = 0 by changing
        _max_vocab = len([1 for k in six.iterkeys(we.vocab) if k in freqs_loaded])
        if eos is not None and eos not in freqs_loaded:
            _max_vocab += 1
        if unk is not None and unk not in freqs_loaded:
            _max_vocab += 1
        max_vocab = _max_vocab if max_vocab is None else min(max_vocab, _max_vocab)

    if max_vocab is not None:
        # index of eos/unk is always smaller than max_vocab so it is safe to
        # call resize
        we.resize(max_vocab)
    return we.vectors, we.vocab
