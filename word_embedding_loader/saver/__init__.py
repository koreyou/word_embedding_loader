# -*- coding: utf-8 -*-
"""
loader module provides actual implementation of the file savers.

.. warning:: This is an internal implementation. API may change without
             notice in the future, so you should use
             :class:`word_embedding_loader.word_embedding.WordEmbedding`
"""

__all__ = ["glove", "numpy", "word2vec_bin", "word2vec_text"]

from word_embedding_loader.saver import glove
from word_embedding_loader.saver import numpy
from word_embedding_loader.saver import word2vec_bin
from word_embedding_loader.saver import word2vec_text
