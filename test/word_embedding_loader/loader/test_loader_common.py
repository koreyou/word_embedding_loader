from __future__ import absolute_import, print_function

import pytest

import word_embedding_loader.loader.word2vec_bin as word2vec_bin
import word_embedding_loader.loader.word2vec_text as word2vec_text
import word_embedding_loader.loader.glove as glove



@pytest.fixture(params=['word2vec_bin_file', 'word2vec_text_file', 'glove_file'])
def context(request):
    if request.param == 'word2vec_bin_file':
        return (word2vec_bin, request.getfixturevalue('word2vec_bin_file'))
    elif request.param == 'word2vec_text_file':
        return (word2vec_text, request.getfixturevalue('word2vec_text_file'))
    elif request.param == 'glove_file':
        return (glove, request.getfixturevalue('glove_file'))


@pytest.mark.parametrize('n, expected', [(2, 2), (3, 3), (5, 3)])
def test_max_vocab(context, n, expected):
    module, f = context
    arr, vocab = module.load(f, max_vocab=n)
    assert len(arr) == expected
    assert len(vocab) == expected
