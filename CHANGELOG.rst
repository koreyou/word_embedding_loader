.. -*- coding: utf-8; -*-


CHANGELOG
=============


v0.2
-------------

* Supports for python 3.4+
* ``WordEmbedding.vocab`` stores words as bytes instead of unicode.
** This allows more consistent loading/saving without needing to care about encoding.
* bugfix:
** building sphinx fails when package is not installed
** issues loading pretrained word2vec GoogleNews-vectors-negative300.bin (#1, #4)

v0.1
-------------

* First release.
* Supports word2vec and glove.
* Documentation using Sphinx.
* CLI interface for converting formats.
