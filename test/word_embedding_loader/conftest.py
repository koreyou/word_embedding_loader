# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import StringIO
import os

import numpy as np
import pytest

import word_embedding_loader.loader.glove as glove
from word_embedding_loader.loader import word2vec_bin
from word_embedding_loader.loader import word2vec_text


@pytest.fixture
def load_glove():
    f = StringIO.StringIO(u"""the 0.418 0.24968 -0.41242 0.1217
, 0.013441 0.23682 -0.16899 0.40951
. 0.15164 0.30177 -0.16763 0.17684""")
    return glove.load(f, None, dtype=np.float32)


@pytest.fixture
def load_word2vec_bin():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(cur_dir, 'word2vec.bin')
    with open(path, 'rb') as f:
        ret = word2vec_bin.load(f, 30)
    return ret


@pytest.fixture
def load_word2vec_text():
    f = StringIO.StringIO(u"""2 3
</s> 0.080054 0.088388 -0.07660
the -1.420859 1.156857 0.744776""")
    return word2vec_text.load(f, None, dtype=np.float32)
