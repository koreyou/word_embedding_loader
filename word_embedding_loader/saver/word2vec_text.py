# -*- coding: utf-8 -*-
u"""
Low level API for saving of word embedding file that was implemented in
`word2vec <https://code.google.com/archive/p/word2vec/>`_, by Mikolov.
This implementation is for word embedding file created with ``-binary 0``
option (the default).
"""
from __future__ import absolute_import, print_function


def _write_line(f, vec, word):
    v_text = u' '.join(map(str, vec))
    # Avoid empty line at the end
    f.write('\n{} {}'.format(word, v_text))


def save(f, arr, vocab):
    """
    Save word embedding file.
    Check :func:`word_embedding_loader.saver.glove.save` for the API.
    """
    f.write('%d %d' % (arr.shape[0], arr.shape[1]))
    for word, idx in vocab:
        _write_line(f, arr[idx], word)
