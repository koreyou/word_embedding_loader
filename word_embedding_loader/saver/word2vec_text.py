# -*- coding: utf-8 -*-
"""
Low level API for saving of word embedding file that was implemented in
`word2vec <https://code.google.com/archive/p/word2vec/>`_, by Mikolov.
This implementation is for word embedding file created with ``-binary 0``
option (the default).
"""
from __future__ import absolute_import, division, print_function, \
    unicode_literals

import six


def _write_line(f, vec, word):
    v_text = ' '.join(map(lambda v: six.u(str(v)), vec))
    # Avoid empty line at the end
    f.write(('\n{} {}'.format(word.decode('utf-8'), v_text)).encode('utf-8'))


def save(f, arr, vocab):
    """
    Save word embedding file.
    Check :func:`word_embedding_loader.saver.glove.save` for the API.
    """
    f.write(('%d %d' % (arr.shape[0], arr.shape[1])).encode('utf-8'))
    for word, idx in vocab:
        _write_line(f, arr[idx], word)
