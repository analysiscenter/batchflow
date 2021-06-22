""" Contains helper functions """
import copy
import functools
import itertools

import numpy as np
try:
    import pandas as pd
except ImportError:
    from . import _fake as pd

from matplotlib import pyplot as plt
from matplotlib import colors as mcolors


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

def show_research(df, layouts=None, titles=None, average_repetitions=False, log_scale=False,
                  rolling_window=None, color=None, **kwargs): # pylint: disable=too-many-branches
    """Show plots given by research dataframe.

    Parameters
    ----------
    df : DataFrame
        Research's results
    layouts : list, optional
        List of strings where each element consists two parts that splited by /. First part is the type
        of calculated value wrote in the "name" column. Second is name of column  with the parameters
        that will be drawn.
    titles : list, optional
        List of titles for plots that defined by layout.
    average_repetitions : bool, optional
        If True, then a separate line will be drawn for each repetition
        else one mean line will be drawn for each repetition.
    log_scale : bool or sequence of bools, optional
        If True, values will be logarithmised.
    rolling_window : int of sequence of ints, optional
        Size of rolling window.
    color: str or sequence of matplotlib.colors, optional
        If str, should be a name of matplotlib colormap,
        colors for plots will be selected from that colormap.
        If sequence of colors, they will be used for plots,
        if sequence length is less, than number of lines to plot,
        colors will be repeated in cycle
        If None (default), `mcolors.TABLEAU_COLORS` sequence is used
    kwargs:
        Additional named arguments directly passed to `plt.subplots`.
        With default parameters:
            - ``figsize = (9 * len(layouts), 7)``
            - ``nrows = 1``
            - ``ncols = len(layouts)``
    """
    if layouts is None:
        layouts = []
        for nlabel, ndf in df.groupby("name"):
            ndf = ndf.drop(['config', 'name', 'iteration', 'repetition'], axis=1).dropna(axis=1)
            for attr in ndf.columns.values:
                layouts.append('/'.join([str(nlabel), str(attr)]))
    titles = layouts if titles is None else titles
    if isinstance(log_scale, bool):
        log_scale = [log_scale] * len(layouts)
    if isinstance(rolling_window, int) or (rolling_window is None):
        rolling_window = [rolling_window] * len(layouts)
    rolling_window = [x if x is not None else 1 for x in rolling_window]

    if color is None:
        color = list(mcolors.TABLEAU_COLORS.keys())
    df_len = len(df['config'].unique())

    if isinstance(color, str):
        cmap = plt.get_cmap(color)
        chosen_colors = [cmap(i/df_len) for i in range(df_len)]
    else:
        chosen_colors = list(itertools.islice(itertools.cycle(color), df_len))

    kwargs = {'figsize': (9 * len(layouts), 7), 'nrows': 1, 'ncols': len(layouts), **kwargs}

    _, ax = plt.subplots(**kwargs)
    if len(layouts) == 1:
        ax = (ax, )

    for i, (layout, title, log, roll_w) in enumerate(list(zip(*[layouts, titles, log_scale, rolling_window]))):
        name, attr = layout.split('/')
        ndf = df[df['name'] == name]
        for (clabel, cdf), curr_color in zip(ndf.groupby("config"), chosen_colors):
            cdf = cdf.drop(['config', 'name'], axis=1).dropna(axis=1).astype('float')
            if average_repetitions:
                idf = cdf.groupby('iteration').mean().drop('repetition', axis=1)
                y_values = idf[attr].rolling(roll_w).mean().values
                if log:
                    y_values = np.log(y_values)
                ax[i].plot(idf.index.values, y_values, label=str(clabel), color=curr_color)
            else:
                for repet, rdf in cdf.groupby('repetition'):
                    rdf = rdf.drop('repetition', axis=1)
                    y_values = rdf[attr].rolling(roll_w).mean().values
                    if log:
                        y_values = np.log(y_values)
                    ax[i].plot(rdf['iteration'].values, y_values,
                               label='/'.join([str(repet), str(clabel)]), color=curr_color)
        ax[i].set_xlabel('iteration')
        ax[i].set_title(title)
        ax[i].legend()
    plt.show()


def print_results(df, layout, average_repetitions=False, sort_by=None, ascending=True, n_last=100):
    """ Show results given by research dataframe.

    Parameters
    ----------
    df : DataFrame
        Research's results
    layout : str
        string where each element consists two parts that splited by /. First part is the type
        of calculated value wrote in the "name" column. Second is name of column  with the parameters
        that will be drawn.
    average_repetitions : bool, optional
        If True, then a separate values will be written
        else one mean value will be written.
    sort_by : str or None, optional
        If not None, column's name to sort.
    ascending : bool, None
        Same as in ```pd.sort_value```.
    n_last : int, optional
        The number of iterations at the end of which the averaging takes place.

    Returns
    -------
        : DataFrame
        Research results in DataFrame, where indices is a config parameters and colums is `layout` values
    """
    columns = []
    data = []
    index = []
    name, attr = layout.split('/')
    ndf = df[df['name'] == name]
    if average_repetitions:
        columns.extend([attr + ' (mean)', attr + ' (std)'])
    else:
        repetition_cols = ['　(repetition {})'.format(i) for i in ndf['repetition'].unique()]
        columns.extend([attr + col_name for col_name in [*repetition_cols, ' (mean)', ' (std)']])

    for config, cdf in ndf.groupby("config"):
        index.append(config)
        cdf = cdf.drop(['config', 'name'], axis=1).dropna(axis=1).astype('float')
        rep = []
        for _, rdf in cdf.groupby('repetition'):
            rdf = rdf.drop('repetition', axis=1)
            rdf = rdf[rdf['iteration'] > rdf['iteration'].max() - n_last]
            rep.append(rdf[attr].mean())
        if average_repetitions:
            data.append([np.mean(rep), np.std(rep)])
        else:
            data.append([*rep, np.mean(rep), np.std(rep)])

    res_df = pd.DataFrame(data=data, index=index, columns=columns)
    if sort_by:
        res_df.sort_values(by=sort_by, ascending=ascending, inplace=True)
    return res_df


def plot_images(images, labels=None, proba=None, ncols=5, classes=None, models_names=None, **kwargs):
    """ Plot images and optionally true labels as well as predicted class proba.
        - In case labels and proba are not passed, just shows images.
        - In case labels are passed and proba is not, shows images with labels.
        - Otherwise shows everything.
    In case the predictions of several models provided, i.e proba is an iterable containing np.arrays,
    shows predictions for every model.

    Parameters
    ----------
    images : np.array
        Batch of images.
    labels : array-like, optional
        Images labels.
    proba: np.array with the shape (n_images, n_classes) or list of such arrays, optional
        Predicted probabilities for each class for each model.
    ncols: int
        Number of images to plot in a row.
    classes: list of strings
        Class names. In case not specified the list [`1`, `2`, .., `proba.shape[1]`] would be assigned.
    models_names: string or list of strings
        Models names. In case not specified and the single model predictions provided will not display any name.
        Otherwise the list [`Model 1`, `Model 2`, ..] is being assigned.
    kwargs : dict
        Additional keyword arguments for plt.subplots().
    """
    if isinstance(models_names, str):
        models_names = (models_names, )
    if not isinstance(proba, (list, tuple)):
        proba = (proba, )
        if models_names is None:
            models_names = ['']
    else:
        if models_names is None:
            models_names = ['Model ' + str(i+1) for i in range(len(proba))]

    # if the classes names are not specified they can be implicitely infered from the `proba` shape,
    if classes is None:
        if proba[0] is not None:
            classes = [str(i) for i in range(proba[0].shape[1])]
        elif labels is None:
            pass
        elif proba[0] is None:
            raise ValueError('Specify classes')

    n_items = len(images)
    nrows = (n_items // ncols) + 1
    fig, ax = plt.subplots(nrows, ncols, **kwargs)
    ax = ax.flatten()
    for i in range(n_items):
        ax[i].imshow(images[i])
        if labels is not None: # plot images with labels
            true_class_name = classes[labels[i]]
            title = 'Real answer: {}'.format(true_class_name)
            if proba[0] is not None: # plot images with labels and predictions
                for j, model_proba in enumerate(proba): # the case of preidctions of several models
                    class_pred = np.argmax(model_proba, axis=1)[i]
                    class_proba = model_proba[i][class_pred]
                    pred_class_name = classes[class_pred]
                    title += '\n {} Prediction: {} with {:.2f}%'.format(models_names[j],
                                                                        pred_class_name, class_proba * 100)
            ax[i].title.set_text(title)
            ax[i].title.set_size(28)
        ax[i].grid(b=None)

    for i in range(n_items, nrows * ncols):
        fig.delaxes(ax[i])


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
