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
        raise ValueError('len(we.vectors) < n (%d < %d)' % (we.vectors, n))
    if we.freqs is None:
        return numpy.average(we.vectors[-n:], axis=0)
    else:
        freqs = sorted(six.iteritems(we.freqs), key=lambda k_v: k_v[1])
        return numpy.average([we.vectors[we.vocab[k]] for k, _ in freqs[:n]],
                             axis=0)


def create_random_vector(size, std, dtype):
    assert std > 0.
    return numpy.random.normal(0., std, (size, )).astype(dtype)
