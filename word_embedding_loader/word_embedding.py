# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import numpy as np

from word_embedding_loader.loader import glove, word2vec_text, word2vec_bin


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
    if glove.check_valid(l0, l1):
        return glove
    elif word2vec_text.check_valid(l0, l1):
        return word2vec_text
    elif word2vec_bin.check_valid(l0, l1):
        return word2vec_bin
    else:
        raise OSError("Invalid format")


class WordEmbedding(object):
    def __init__(self, path, dtype=np.float32, max_vocab=None, format=None,
                 keep_order=False, encoding='utf-8', unicode_errors='strict'):
        """
        Load pretrained word embedding from a file.

        Args:
            path:
            format:

        Returns:
            numpy.ndarray: Word embedding representation vectors
            dict: Mapping from words to vector indices.
            numpy.ndarray or None: Word counts, indexed by the dict.
                It will be None if word counts information was not provided in the
                file.

        """
        with open(path, mode='r') as f:
            if format is None:
                loader = _classify_format(f)
            else:
                if format == 'glove':
                    loader = glove
                elif format == 'word2vec_bin':
                    loader = word2vec_bin
                elif format == 'word2vec_text':
                    loader = word2vec_text
                else:
                    raise NameError('Unknown format "%s"' % format)
            arr, vocab = loader.load(
                f, max_vocab=max_vocab, dtype=dtype, keep_order=keep_order,
                unicode_errors=unicode_errors, encoding=encoding)
        self.vectors = arr
        self.vocab = vocab

    def save(self, path, format):
        pass

    def __len__(self):
        return len(self.vectors)

    @property
    def n_features(self):
        return self.vectors.shape[1]

