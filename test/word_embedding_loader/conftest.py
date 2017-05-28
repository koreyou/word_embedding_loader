# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import StringIO
import os

import pytest


@pytest.fixture
def glove_file():
    f = StringIO.StringIO(u"""the 0.418 0.24968 -0.41242 0.1217
, 0.013441 0.23682 -0.16899 0.40951
日本語 0.15164 0.30177 -0.16763 0.17684""".encode('utf-8'))
    yield f
    f.close()


@pytest.fixture
def word2vec_bin_file():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(cur_dir, 'word2vec.bin')
    with open(path, 'rb') as f:
        yield f


@pytest.fixture
def word2vec_text_file():
    f = StringIO.StringIO(u"""3 2
</s> 0.080054 0.088388
the -1.420859 1.156857
日本語 -0.16799 0.10951""".encode('utf-8'))
    yield f
    f.close()


@pytest.fixture
def vocab_file():
    f = StringIO.StringIO(u"""</s> 0
the 1061396
日本語 593677""".encode('utf-8'))
    yield f
    f.close()
