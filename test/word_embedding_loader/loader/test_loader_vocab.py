# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

from collections import OrderedDict

from word_embedding_loader.loader import vocab


def test_load_vocab(vocab_file):
    expected = OrderedDict(((u"</s>", 0), (u"日本語", 593677)))
    assert vocab.load_vocab(vocab_file) == expected
