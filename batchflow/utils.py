""" Contains helper functions """
import sys
import copy
import math
import functools
import tqdm

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors

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

def show_research(df, layout=None, average_repetitions=False, log_scale=False,
                  rolling_window=None, color=None, scale=(9, 7)): # pylint: disable=too-many-branches
    """Show plots given by research dataframe.

    Parameters
    ----------
    df : DataFrame
        Research's results
    layout : list, optional
        List of strings where each element consists two parts that splited by /. First part is the type
        of calculated value wrote in the "name" column. Second is name of column  with the parameters
        that will be drawn.
    average_repetitions : bool, optional
        If True, then a separate line will be drawn for each repetition
        else one mean line will be drawn for each repetition.
    log_scale : bool or sequence of bools, optional
        If True, values will be logarithmised.
    rolling_window : int of sequence of ints, optional
        Size of rolling window.
    color: sequence of matplotlib.colors, optional
        Colors for plots would be randomly sampled from given set.
    scale: tuple, default: (9, 7)
        Scaling factors for the figure.
    """
    if layout is None:
        layout = []
        for nlabel, ndf in df.groupby("name"):
            ndf = ndf.drop(['config', 'name', 'iteration', 'repetition'], axis=1).dropna(axis=1)
            for attr in ndf.columns.values:
                layout.append('/'.join([str(nlabel), str(attr)]))
    if isinstance(log_scale, bool):
        log_scale = [log_scale] * len(layout)
    if isinstance(rolling_window, int) or (rolling_window is None):
        rolling_window = [rolling_window] * len(layout)
    rolling_window = [x if x is not None else 1 for x in rolling_window]

    if color is None:
        color = list(mcolors.CSS4_COLORS.keys())
    df_len = len(df['config'].unique())
    replace = not len(color) > df_len
    chosen_colors = np.random.choice(color, replace=replace, size=df_len)

    _, ax = plt.subplots(1, len(layout), figsize=(scale[0] * len(layout), scale[1]))
    if len(layout) == 1:
        ax = (ax, )

    for i, (title, log, roll_w) in enumerate(list(zip(*[layout, log_scale, rolling_window]))):
        name, attr = title.split('/')
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
        columns.extend([name + '_mean', name + '_std'])
    else:
        columns.extend([name + '_' + str(i) for i in [*ndf['repetition'].unique(), 'mean', 'std']])
    for config, cdf in ndf.groupby("config"):
        index.append(config)
        cdf = cdf.drop(['config', 'name'], axis=1).dropna(axis=1).astype('float')
        if average_repetitions:
            idf = cdf.groupby('iteration').mean().drop('repetition', axis=1)
            max_iter = idf.index.max()
            idf = idf[idf.index > max_iter - n_last]
            data.append([idf[attr].mean(), idf[attr].std()])
        else:
            rep = []
            for _, rdf in cdf.groupby('repetition'):
                rdf = rdf.drop('repetition', axis=1)
                max_iter = rdf['iteration'].max()
                rdf = rdf[rdf['iteration'] > max_iter - n_last]
                rep.append(rdf[attr].mean())
            data.append([*rep, np.mean(rep), np.std(rep)])

    res_df = pd.DataFrame(data=data, index=index, columns=columns)
    if sort_by:
        res_df.sort_values(by=sort_by, ascending=ascending, inplace=True)
    return res_df


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

def save_data_to(what, where, **kwargs):
    """ Store data to specified locations

    Parameters
    ----------
    what : value or a list of values

    where : NamedExpression, array or a list of them

    kwargs
        arguments to be passed into a NamedExpression
    """
    if not isinstance(where, (tuple, list)):
        where = [where]
        if isinstance(what, (tuple, list)):
            what = [what]
    if not isinstance(what, (tuple, list)):
        what = [what]

    if len(where) != len(what):
        raise ValueError("The lengths of outputs and saving locations mismatch")

    for i, var in enumerate(where):
        item = what[i]
        if isinstance(var, NamedExpression):
            var.set(item, **kwargs)
        elif isinstance(var, np.ndarray):
            var[:] = item
        else:
            where[i] = item
