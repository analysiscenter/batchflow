""" Contains helper functions """
import sys
import copy
import math
from functools import wraps
import tqdm

from .named_expr import NamedExpression, eval_expr



def partialmethod(func, *frozen_args, **frozen_kwargs):
    """Wrap a method with partial application of given positional and keyword
    arguments.

    Parameters
    ----------
    func : callable
        A method to wrap.
    frozen_args : misc
        Fixed positional arguments.
    frozen_kwargs : misc
        Fixed keyword arguments.

    Returns
    -------
    method : callable
        Wrapped method.
    """
    @wraps(func)
    def method(self, *args, **kwargs):
        """Wrapped method."""
        return func(self, *frozen_args, *args, **frozen_kwargs, **kwargs)
    return method

def copy1(data):
    """ Copy data exactly 1 level deep """
    if isinstance(data, tuple):
        out = tuple(_copy1_list(data))
    elif isinstance(data, list):
        out = _copy1_list(data)
    elif isinstance(data, dict):
        out = _copy1_dict(data)
    else:
        out = copy.copy(data)
    return out

def _copy1_list(data):
    return [copy.copy(item) for item in data]

def _copy1_dict(data):
    return dict((key, copy.copy(item)) for key, item in data.items())


def create_bar(bar, batch_size, n_iters, n_epochs, drop_last, length):
    """ Create progress bar with desired number of total iterations."""
    if n_iters is not None:
        total = n_iters
    elif n_epochs is None:
        total = sys.maxsize
    elif drop_last:
        total = length // batch_size * n_epochs
    else:
        total = math.ceil(length * n_epochs / batch_size)

    if callable(bar):
        progressbar = bar(total=total)
    elif bar == 'n':
        progressbar = tqdm.tqdm_notebook(total=total)
    else:
        progressbar = tqdm.tqdm(total=total)
    return progressbar


def update_bar(bar, bar_desc, **kwargs):
    """ Update bar with description and one step."""
    if bar_desc:
        if callable(bar_desc) and not isinstance(bar_desc, NamedExpression):
            desc = bar_desc()
        else:
            try:
                desc = eval_expr(bar_desc, **kwargs)
                desc = str(desc)
            except (LookupError, ValueError):
                desc = None
        bar.set_description(desc)
    bar.update(1)
