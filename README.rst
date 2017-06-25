.. -*- coding: utf-8; -*-

Loaders and savers for different implentations of `word embedding <https://en.wikipedia.org/wiki/Word_embedding>`_. The motivation of this project is that it is cumbersome to write loaders for different pretrained word embedding files. This project provides a simple interface for loading pretrained word embedding files in different formats.

.. code:: python

   from word_embedding_loader import WordEmbedding

   # it will automatically determine format from content
   wv = WordEmbedding.load('path/to/embedding.bin')

   # This project provides minimum interface for word embedding
   print wv.vectors[wv.vocab[u'is']]

   # Modify and save word embedding file with arbitrary format
   wv.save('path/to/save.txt', 'word2vec', binary=False)


This project currently supports following formats:

* `GloVe <https://nlp.stanford.edu/projects/glove/>`_, Global Vectors for Word Representation, by Jeffrey Pennington, Richard Socher, Christopher D. Manning from Stanford NLP group.
* `word2vec <https://code.google.com/archive/p/word2vec/>`_, by Mikolov.
    - text (create with ``-binary 0`` option (the default))
    - binary (create with ``-binary 1`` option)
* `gensim <https://radimrehurek.com/gensim/>`_ 's ``models.word2vec`` module (coming)
* original HDFS format: a performance centric option for loading and saving word embedding (coming)


Sometimes, you want combine an external program with word embedding file of your own choice. This project also provides a simple executable to convert a word embedding format to another.

.. code:: bash

   # it will automatically determine the format from the content
   word-embedding-loader convert -t glove test/word_embedding_loader/word2vec.bin test.bin

   # Get help for command/subcommand
   word-embedding-loader --help
   word-embedding-loader convert --help


Development
============

This project us Cython to build some modules, so you need Cython for development.

```bash
pip install -r requirements.txt
```

If environment variable ``DEVELOP_WE`` is set, it will try to rebuild ``.pyx`` modules.

```bash
DEVELOP_WE=1 python setup.py test
```
