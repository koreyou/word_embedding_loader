# -*- coding: utf-8 -*-
u"""
Low level API for saving of word embedding file that was implemented in
`GloVe <https://nlp.stanford.edu/projects/glove/>`_, Global Vectors for Word
Representation, by Jeffrey Pennington, Richard Socher, Christopher D. Manning
from Stanford NLP group.
"""
from __future__ import absolute_import, print_function


def _write_line(f, vec, word, encoding, errors):
    v_text = u' '.join(map(str, vec))
    f.write('{} {}'.format(word.encode(encoding, errors=errors), v_text))


def save(f, arr, vocab, counts=None, encoding='utf-8', unicode_errors='strict'):
    """
    Save word embedding file.

    Args:
        f (File): File to write the vectors. File should be open for writing
            ascii.
        arr (numpy.array): Numpy array with ``float`` dtype.
        vocab (dict): Mapping from words (``unicode``) to ``arr`` index
            (``int``).
        counts (dict or None): Mapping from words (``unicode``) to counts
            (``int``). If ``None``, orders of ``arr`` is used.
        encoding (str): Encoding of the input file as defined in ``codecs``
            module of Python standard library.
        unicode_errors (str): Set the error handling scheme. The default error
            handler is 'strict' meaning that encoding errors raise ValueError.
            Refer to ``codecs`` module for more information.
    """
    if counts is None:
        itr = sorted(vocab.iteritems(), key=lambda (k, v): v)
    else:
        itr = sorted(
            vocab.iteritems(), key=lambda (k, v): counts[k], reverse=True)

    itr = iter(itr)
    # Avoid empty line at the end
    word, idx = next(itr)
    _write_line(f, arr[idx], word, encoding, unicode_errors)
    for word, idx in itr:
        f.write('\n')
        _write_line(f, arr[idx], word, encoding, unicode_errors)
