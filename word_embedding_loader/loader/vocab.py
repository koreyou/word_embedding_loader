# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

from collections import OrderedDict

def load_vocab(fin):
    vocab = OrderedDict()
    for line in fin:
        v, c = line.strip().split()
        vocab[v] = int(c)
    return vocab
