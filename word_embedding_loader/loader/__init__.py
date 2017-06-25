u"""
loader module provides actual implementation of the file loaders.

.. warning:: This is an internal implementation. API may change without
             notice in the future, so you should use
             :class:`word_embedding_loader.word_embedding.WordEmbedding`
"""

__all__ = ["glove", "vocab", "word2vec_bin", "word2vec_text"]

from word_embedding_loader.loader import glove
from word_embedding_loader.loader import vocab
from word_embedding_loader.loader import word2vec_bin
from word_embedding_loader.loader import word2vec_text
