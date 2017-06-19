# -*- coding: utf-8 -*-
from collections import OrderedDict

import click

from word_embedding_loader import word_embedding


# Each value is (description, format string, is_binary) tuple
_input_choices = OrderedDict([
    ('auto', ('Determine from content', None, False))
])

_output_choices = OrderedDict((
    ('glove', ("GloVe by Stanford NLP group.", 'glove', False)),
    ('word2vec', ("alias of word2vec-text", 'word2vec', False)),
    ('word2vec-text', ("word2vec (by Mikolov et al.) with -binary 0 option.", 'word2vec', False)),
    ('word2vec-binary', ("word2vec (by Mikolov et al.) with -binary 1 option.", 'word2vec', True))
))
_input_choices.update(_output_choices)


@click.group()
def cli():
    pass


@cli.command()
@click.argument('inputfile', type=click.Path(exists=True))
@click.argument('outputfile', type=click.Path())
@click.option('-t', '--to-format', type=click.Choice(_output_choices.keys()),
              help='Target format')
@click.option('-f', '--from-format', type=click.Choice(_input_choices.keys()),
              default='auto', help='Source format. It will guess format from content if not given.')
def convert(outputfile, inputfile, to_format, from_format):
    """
    Convert pretrained word embedding file in one format to another.
    """
    emb = word_embedding.WordEmbedding.load(
        inputfile, format=_input_choices[from_format][1],
        binary=_input_choices[from_format][2])
    emb.save(outputfile, format=_output_choices[to_format][1],
             binary=_output_choices[to_format][2])


def _echo_format_result(name):
    click.echo("{}: {}".format(name, _input_choices[name][0]))


@cli.command()
@click.argument('inputfile', type=click.File())
def check_format(inputfile):
    """
    Check format of inputfile.
    """
    t = word_embedding.classify_format(inputfile)
    if t == word_embedding._glove:
        _echo_format_result('glove')
    elif t == word_embedding._word2vec_bin:
        _echo_format_result('word2vec-binary')
    elif t == word_embedding._word2vec_text:
        _echo_format_result('word2vec-text')
    else:
        assert not "Should not get here!"


@cli.command()
def list():
    """
    List available format.
    """
    choice_len = max(map(len, _input_choices.keys()))
    tmpl = "  {:<%d}: {}\n" % choice_len
    text = ''.join(map(
        lambda (k, v): tmpl.format(k, v[0]), _input_choices.iteritems()))
    click.echo(text)