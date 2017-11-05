# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, \
    unicode_literals

import numpy
import six

UNK_ID_CANDIDATES = [b'<unk>', b'<UNK>']


def infer_unk_id(vocab):
    """

    Args:
        vocab (dict): Mapping from words (bytes) to index (int)

    Returns:
        bytes: "unk" id (as a word in ``vocab``)

    """
    candidates = []
    for c in UNK_ID_CANDIDATES:
        if c in vocab:
            candidates.append(c)
    if len(candidates) > 1:
        raise ValueError(b'Ambiguous unk id found (' + bytes(candidates) + b')')
    if len(candidates) == 0:
        return None
    return candidates[0]


def create_unk_least_common(we, n):
    """

    Args:
        we (~WordEmbedding):

    Returns:

    """
    if len(we.vectors) < n:
        raise ValueError('len(we.vectors) < n (%d < %d)' % (len(we.vectors), n))
    if we.freqs is None:
        return numpy.average(we.vectors[-n:], axis=0)
    else:
        freqs = sorted(six.iteritems(we.freqs), key=lambda k_v: k_v[1])
        return numpy.average([we.vectors[we.vocab[k]] for k, _ in freqs[:n]],
                             axis=0)


def create_random_vector(size, std, dtype, seed=None):
    assert std > 0.
    numpy.random.seed(seed)
    return numpy.random.normal(0., std, (size, )).astype(dtype)


def move_vocab(vocab, vectors, key, index):
    """
    Move the particular word to specified index.

    .. warning:: This function has side effect on ``vocab``.

    Args:
        vocab (dict): Mapping from word (bytes) to index (int)
        vectors (numpy.ndarray): Word embedding vectors
        key (bytes): New word to insert to move
        index (int): Index to which ``vec`` is inserted

    Returns:
        dict: Modified vocab file
        numpy.ndarray: Modified vectors
        int or None: Original index of the key in vocab

    """
    if key not in vocab:
        raise KeyError(b'key "' + key + b'" does no exist in vocab.')
    vec = vectors[vocab[key]]
    vectors = numpy.delete(vectors, vocab[key], axis=0)
    vectors = numpy.insert(vectors, index, vec, axis=0)

    old_ind = vocab[key]
    if old_ind == index:
        return vocab, vectors, index
    elif old_ind > index:
        for k in vocab.keys():
            if vocab[k] >= index and vocab[k] < old_ind:
                vocab[k] += 1
    else:
        for k in vocab.keys():
            if vocab[k] > old_ind and vocab[k] <= index:
                vocab[k] -= 1
    vocab[key] = index
    return vocab, vectors, old_ind


def insert_vocab(vocab, vectors, key, index, vec):
    """
    Insert new word to word embedding.

    .. warning:: This function has side effect on ``vocab``.

    Args:
        vocab (dict): Mapping from word (bytes) to index (int)
        vectors (numpy.ndarray): Word embedding vectors
        key (bytes): New word to insert to move
        index (int): Index to which ``vec`` is inserted
        vec (numpy.ndarray): A vector to insert

    Returns:
        dict: Modified vocab file
        numpy.ndarray: Modified vectors

    """
    assert vec.ndim == 1 or (vec.ndim == 2 and vec.shape[0] == 1)
    assert vec.shape[-1] == vectors.shape[1]
    if key in vocab:
        raise KeyError(b'key "' + key + b'"%s already exists in vocab.')

    for k in vocab.keys():
        if vocab[k] >= index:
            vocab[k] += 1

    vocab[key] = index
    vectors = numpy.insert(vectors, index, vec, axis=0)
    return vocab, vectors


def remove_vocab(vocab, vectors, key):
    """
    Move the particular word to specified index.

    .. warning:: This function has side effect on ``vocab``.

    Args:
        vocab (dict): Mapping from word (bytes) to index (int)
        vectors (numpy.ndarray): Word embedding vectors
        key (bytes): New word to insert to move
        index (int): Index to which ``vec`` is inserted

    Returns:
        dict: Modified vocab file
        numpy.ndarray: Modified vectors
        int or None: Original index of the key in vocab

    """
    if key not in vocab:
        raise KeyError(b'key "' + key + b'" does no exist in vocab.')
    vectors = numpy.delete(vectors, vocab[key], axis=0)

    old_ind = vocab[key]
    for k in vocab.keys():
        if vocab[k] > old_ind:
            vocab[k] -= 1
    del vocab[key]
    return vocab, vectors, old_ind
