""" Contains parser. On importing from executable script, `add_tokens` is executed.
See :meth:`.parser.add_tokens` for details.
"""
import os
import inspect

from .parser import *

MODULE = os.environ.get('TOKENS_MODULE') or 'tf'

FRAME = inspect.currentframe()
for i in range(30):
    FRAME = FRAME.f_back
    calling_locals = FRAME.f_locals

    name_ = calling_locals.get('__name__')
    if name_ == '__main__':
        break

add_tokens(calling_locals, module=MODULE)
