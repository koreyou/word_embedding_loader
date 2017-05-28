# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function

import warnings


class ParseError(Exception):
    pass


class ParseWarning(Warning):
    pass


def parse_warn(message):
    warnings.warn(message, ParseWarning)
