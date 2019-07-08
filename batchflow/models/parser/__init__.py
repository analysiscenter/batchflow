""" Contains parser. On importing from executable script, `add_tokens` is executed.
See :meth:`.parser.add_tokens` for details.
"""
import os
import inspect

from .parser import *

module = os.environ.get('TOKENS_MODULE') or 'tf'

frame = inspect.currentframe()
for i in range(30):
    frame = frame.f_back
    calling_locals = frame.f_locals

    name_ = calling_locals.get('__name__')
    if name_ == '__main__':
        break

add_tokens(calling_locals, module=module)
