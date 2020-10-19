""" Contains helper functions """
import os
import re
import json
import copy
import functools
import itertools

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors

import ipykernel
import requests
from notebook.notebookapp import list_running_servers
from nbconvert import PythonExporter
from pylint import epylint as lint

from .named_expr import NamedExpression


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
        chosen_colors = itertools.cycle(color)

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


def in_notebook():
    """ Return True if in Jupyter notebook and False otherwise. """
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True
        if shell == 'TerminalInteractiveShell':
            return False
        return False
    except NameError:
        return False

def get_notebook_path():
    """ Return the full absolute path of the current Jupyter notebook,
    for example, `/path/path/path/My_notebook_title.ipynb`.

    If run outside Jupyter notebook, returns None.
    """
    if not in_notebook():
        return None

    kernel_id = re.search('kernel-(.*).json',
                          ipykernel.connect.get_connection_file()).group(1)
    servers = list_running_servers()
    for server in servers:
        response = requests.get(requests.compat.urljoin(server['url'], 'api/sessions'),
                                params={'token': server.get('token', '')})
        for params in json.loads(response.text):
            if params['kernel']['id'] == kernel_id:
                relative_path = params['notebook']['path']
                return os.path.join(server['notebook_dir'], relative_path)
    return None

def get_notebook_name():
    """ Return the title of the current Jupyter notebook without base directory and extension,
    for example, `My_notebook_title`.

    If run outside Jupyter notebook, returns None.
    """
    if not in_notebook():
        return None

    return os.path.splitext(get_notebook_path())[0].split('/')[-1]


def pylint_notebook(path=None, options='', printer=print, ignore_comments=True, ignore_codes=None,
                    keep_script=False, return_report=False):
    """ Run pylint on entire Jupyter notebook.
    Under the hood, the notebook is converted to regular `.py` script,
    special IPython commands like magics removed, and then pylint is executed.

    If run outside Jupyter notebook, returns 1.

    Parameters
    ----------
    path : str, optional
        Path to run linter on. If not provided, the callee notebook is linted.
    options : str
        Additional flags for linter execution, for example, the pylint configuration options.
    printer : callable
        Method for displaying results.
    ignore_comments : bool
        Whether to ignore markdown cells and comments in code.
    ignore_codes : sequence
        Pylint errors to ignore.
        By default, `invalid-name`, `import-error` and `wrong-import-position` are disabled.
    keep_script : bool
        Whether to keep temporal `.py` file after command execution.
    return_report : bool
        If True, then this function returns the string representation of produced report.
        If False, then 0 is returned.
    """
    if not in_notebook():
        return 1

    path = path or get_notebook_path()
    options = options if options.startswith(' ') else ' ' + options
    ignore_codes = ignore_codes or ['invalid-name', 'import-error', 'wrong-import-position']

    # Convert the notebook contents to raw string without outputs
    code, _ = PythonExporter().from_filename(path)

    # Unwrap code lines from line/cell magics
    code_list = []
    cell_codes, cell_counter = [], 0
    cell_code_lines, cell_code_counter = [], 1

    for line in code.split('\n'):
        # Line magics: remove autoreload
        if line.startswith('get_ipython().run_line_magic'):
            if 'autoreload' in line:
                line = ''
            else:
                line = line[line.find(',')+3:-2]

        # Cell magics: contain multiple lines
        if line.startswith('get_ipython().run_cell_magic'):
            line = line[line.find(',')+1:]
            line = line[line.find(',')+3:-2]

            lines = line.split('\\n')
        else:
            lines = [line]

        # Update all the containers
        for part in lines:
            code_list.append(part)
            cell_codes.append(cell_counter)
            cell_code_lines.append(cell_code_counter)
            cell_code_counter += 1

        if line.startswith('# In['):
            cell_counter += 1
            cell_code_counter = 0

    code = '\n'.join(code_list)

    # Create temporal file with code, run pylint on it
    temp_name = os.path.splitext(path)[0] + '.py'
    with open(temp_name, 'w') as temp_file:
        temp_file.write(code)

    pylint_stdout, pylint_stderr = lint.py_run(temp_name + options, return_std=True)

    errors = pylint_stderr.getvalue()
    report = pylint_stdout.getvalue()
    if errors:
        printer('Errors \n', errors)

    # Create a better repr of pylint report: remove markdown-related warnings
    report_ = []
    for error_line in report.split('\n'):
        if temp_name in error_line:
            error_line = error_line.replace(temp_name, 'nb')
            code_line_number = int(error_line.split(':')[1])
            code_line = code_list[code_line_number - 1]

            # Ignore markdown and comments
            if ignore_comments and code_line.startswith('#'):
                continue

            # Ignore codes
            if sum(code in error_line for code in ignore_codes):
                continue

            # Create report message
            cell_number = cell_codes[code_line_number - 1]
            cell_code_number = cell_code_lines[code_line_number - 1] - 1
            error_code = error_line[error_line.find('(')+1 : error_line.find('(')+6]
            error_msg = error_line[error_line.find(')')+2:]

            report_msg = f'Cell {cell_number}, line {cell_code_number}, error code {error_code}:'
            report_msg += f'\nPylint message: {error_msg}\nCode line   ::: {code_line}\n'

            report_.append(report_msg)

        if 'rated' in error_line:
            report_.insert(0, error_line.strip(' '))
            report_.insert(1, '-' * (len(error_line) - 1))
            report_.insert(2, '')

    printer('\n'.join(report_))

    # Cleanup
    if not keep_script:
        os.remove(temp_name)

    if return_report:
        return '\n'.join(report_)
    return 0
