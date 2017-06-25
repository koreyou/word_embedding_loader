# -*- coding: utf-8 -*-
u"""
Low level API for saving of word embedding file that was implemented in
`GloVe <https://nlp.stanford.edu/projects/glove/>`_, Global Vectors for Word
Representation, by Jeffrey Pennington, Richard Socher, Christopher D. Manning
from Stanford NLP group.
"""
from __future__ import absolute_import, print_function


def _write_line(f, vec, word):
    v_text = ' '.join(map(str, vec))
    f.write('{} {}'.format(word, v_text))


def save(f, arr, vocab):
    """
    Save word embedding file.

    Args:
        f (File): File to write the vectors. File should be open for writing
            ascii.
        arr (numpy.array): Numpy array with ``float`` dtype.
        vocab (iterable): Each element is pair of a word (``str``) and ``arr``
            index (``int``). Word should be encoded to str apriori.
    """
    itr = iter(vocab)
    # Avoid empty line at the end
    word, idx = next(itr)
    _write_line(f, arr[idx], word)
    for word, idx in itr:
        f.write('\n')
        _write_line(f, arr[idx], word)
