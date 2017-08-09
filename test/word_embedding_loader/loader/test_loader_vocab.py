# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, \
    unicode_literals

from collections import OrderedDict

from word_embedding_loader.loader import vocab


def test_load_vocab(vocab_file):
    expected = OrderedDict(((b"</s>", 0), ("日本語".encode('utf-8'), 593677)))
    assert vocab.load_vocab(vocab_file) == expected
