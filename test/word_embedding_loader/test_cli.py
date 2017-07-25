# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
# Do NOT use unicode_literals; let click handle unicode decoding

import pytest
from click.testing import CliRunner

from word_embedding_loader import cli


def test_cli_list():
    runner = CliRunner()
    result = runner.invoke(cli.cli, ['list'])
    assert result.exit_code == 0
    assert 'GloVe' in result.output


def test_cli_check_format(word2vec_bin_file_path):
    runner = CliRunner()
    result = runner.invoke(cli.cli, ['check_format', word2vec_bin_file_path])
    assert result.exit_code == 0
    assert 'word2vec' in result.output


@pytest.fixture(params=[
    ['--to-format', 'glove'],
    ['--from-format', 'word2vec-binary'],
    ])
def test_cli_check_convert(params, word2vec_bin_file_path, tmpdir):
    p = tmpdir.mkdir("test_cli_check_convert").join("out.txt")
    params = ['convert'] + params + [word2vec_bin_file_path, p.basename]
    runner = CliRunner()
    result = runner.invoke(cli.cli, params)
    assert result.exit_code == 0


@pytest.fixture(params=[
    ['--to-format', 'glove-bin'],
    ['--from-format', 'glove-bin'],
    ])
def test_cli_check_convert_fail(params, word2vec_bin_file_path, tmpdir):
    p = tmpdir.mkdir("test_cli_check_convert").join("out.txt")
    params = ['convert'] + params + [word2vec_bin_file_path, p.basename]
    runner = CliRunner()
    result = runner.invoke(cli.cli, params)
    assert result.exit_code != 0
