# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

from collections import OrderedDict

def load_vocab(fin, encoding='utf-8', errors='strict'):
    vocab = OrderedDict()
    for line in fin:
        v, c = line.strip().split()
        if encoding is not None:
            v = v.decode(encoding, errors=errors)
        vocab[v] = int(c)
    return vocab
