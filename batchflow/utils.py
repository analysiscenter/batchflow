""" Contains helper functions """
import sys
import copy
import math
from functools import wraps
import tqdm

import numpy as np
from matplotlib import pyplot as plt

from .named_expr import NamedExpression, eval_expr


def is_iterable(obj):
    """ Check if an object is a sequence """
    if isinstance(obj, str):
        return False
    try:
        iter(obj)
    except TypeError:
        return False
    return True


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


def plot_results_by_config(results, variables, figsize=None, layout=None, **kwargs):
    """
    Given results from Research.run() draws plots of specified variables for all configs

    Parameters
    ----------
    results : pandas.DataFrame
        results produced by Research.run()
    variables : tuple, dict or sequence of tuples
        variables to plot and names of functions/pipelines they come from.
        if tuple, it is a pair of strings (source_name, variable name)
        if dict, source names are keys and variable names are values: {source_name: variable name, ...}
        if sequence, it is sequence of pairs (source_name, variable name)
    figsize : tuple or None
        figsize to pass to matplotlib. If None (default value) figsize is set to (x, y),
        where x = (5 * number of variables), y = (5 * number of configs in `results`)
    layout: 'flat', 'square' or None
        plot arranging strategy when only one variable is needed (default: None, plots are arranged vertically)
    """
    if isinstance(variables, dict):
        variables = variables.items()
    elif len(variables) == 2 and isinstance(variables[0], str):
        variables = (variables,)

    gbc = results.groupby('config')
    n_configs = len(gbc)
    n_vars = len(variables)

    n_h, n_v = n_vars, n_configs

    if n_vars == 1:
        if layout == 'flat':
            n_h, n_v = n_configs, 1
        if layout == 'square':
            n_h = int(np.sqrt(n_configs))
            n_v = np.ceil(n_configs / n_h).astype(int)

    if figsize is None:
        figsize = (n_h * 5, n_v * 5)

    _, axs = plt.subplots(n_v, n_h, figsize=figsize)
    axs = axs.flatten() if isinstance(axs, np.ndarray) else (axs,)
    for x, (config, df) in enumerate(gbc):
        for y, (source, val) in enumerate(variables):
            ax = axs[n_vars * x + y]

            cols = ['repetition', 'cv_split'] if 'cv_split' in df.columns else 'repetition'

            res = (df[df['name'] == source]
                   .pivot_table(index='iteration', columns=cols, values=val)
                   .rename(columns=lambda s: 'rep ' + str(s), level=0))

            if 'cv_split' in df.columns:
                res = res.rename(columns=lambda s: 'split ' + str(s), level=1)

            res.plot(ax=ax, **kwargs)
            ax.set_title(config + ' ' + source)
            ax.set_xlabel('Iteration')
            ax.set_ylabel(val.replace('_', ' ').capitalize())
            ax.grid(True)
            ax.legend()


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
    current_iter = bar.n
    if bar_desc:
        if callable(bar_desc) and not isinstance(bar_desc, NamedExpression):
            desc = bar_desc()

        if current_iter == 0:
            # During the first iteration we can't get items from empty containers (lists, dicts, etc)
            try:
                desc = eval_expr(bar_desc, **kwargs)
                desc = str(desc)
            except LookupError:
                desc = None
        else:
            desc = eval_expr(bar_desc, **kwargs)
            desc = str(desc)
        bar.set_description(desc)
    bar.update(1)
