from __future__ import absolute_import, print_function
import os
import StringIO

import pytest
import numpy as np
from numpy.testing import assert_allclose

from word_embedding_loader import word_embedding
from word_embedding_loader.loader import glove, word2vec_text, word2vec_bin


def test__get_two_lines():
    f = StringIO.StringIO(u"""2 3
</s> 0.080054 0.088388 -0.07660
the -1.420859 1.156857 0.744776""")
    l0, l1 = word_embedding._get_two_lines(f)
    assert l0 == u"2 3\n"
    assert l1 == u"</s> 0.080054 0.088388 -0.07660\n"


class TestClassifyFormat:
    def test__classify_format_glove(self, glove_file):
        assert word_embedding._classify_format(glove_file) == glove

    def test__classify_format_word2vec_bin(self, word2vec_bin_file):
        assert word_embedding._classify_format(word2vec_bin_file) == word2vec_bin

    def test__classify_format_word2vec_text(self, word2vec_text_file):
        assert word_embedding._classify_format(word2vec_text_file) == word2vec_text
