""" Contains helper functions """
import copy
import functools

import numpy as np


def is_iterable(obj):
    """ Check if an object is a sequence """
    if isinstance(obj, str):
        return False
    try:
        iter(obj)
    except TypeError:
        return False
    return True


def to_list(obj):
    """Cast an object to a list. Almost identical to `list(obj)` for 1-D
    objects, except for `str`, which won't be split into separate letters but
    transformed into a list of a single element.
    """
    return np.array(obj).ravel().tolist()


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
    @functools.wraps(func)
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

def save_data_to(data, dst, **kwargs):
    """ Store data to a given destination

    Parameters
    ----------
    data : value or a list of values

    dst : NamedExpression, array or a list of them

    kwargs
        arguments to be passed into a NamedExpression
    """
    from .named_expr import NamedExpression
    if not isinstance(dst, (tuple, list)):
        dst = [dst]
        if isinstance(dst, (tuple, list)):
            data = [data]
    if not isinstance(data, (tuple, list)):
        data = [data]

    if len(dst) != len(data):
        raise ValueError("The lengths of outputs and saving locations mismatch")

    for i, var in enumerate(dst):
        item = data[i]
        if isinstance(var, NamedExpression):
            var.set(item, **kwargs)
        elif isinstance(var, np.ndarray):
            var[:] = item
        else:
            dst[i] = item


def read_data_from(src, **kwargs):
    """ Read data from a given source

    Parameters
    ----------
    src : NamedExpression, array or a list of them

    kwargs
        arguments to be passed into a NamedExpression
    """
    from .named_expr import NamedExpression
    if not isinstance(src, (tuple, list)):
        src_ = [src]
        data = [None]
    else:
        src_ = src
        data = [None] * len(src)

    for i, var in enumerate(src_):
        if isinstance(var, NamedExpression):
            data[i] = var.get(**kwargs)
        else:
            data[i] = var

    if isinstance(src, (tuple, list)):
        data = type(src)(data)
    else:
        data = data[0]

    return data
