""" Auxilary functions """
import os
import glob
import shutil
import logging
import hashlib
import itertools
import json
from collections import OrderedDict
from copy import deepcopy
import dill
from tqdm import tqdm_notebook
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def to_list(value):
    return value if isinstance(value, list) else [value]

def count_startswith(seq, name):
    return sum(1 for item in seq if item.startswith(name))

def get_metrics(pipeline, metrics_var, metrics_name, *args, agg='mean', **kwargs):
    """ Function to evaluate metrics. """
    metrics_name = metrics_name if isinstance(metrics_name, list) else [metrics_name]
    metrics = pipeline.get_variable(metrics_var).evaluate(metrics_name, *args, agg=agg, **kwargs)
    values = [metrics[name] for name in metrics_name]
    if len(values) == 1:
        return values[0]
    return values

def convert_research_results(research_name, new_name=None, bar=True):
    """ Convert research results from old format to the new. Only results will be transformed, old research can not be
    converted to load by new Research version. """
    # Copy research if needed
    if new_name is not None:
        shutil.copytree(research_name, new_name)
        research_name = new_name

    # Move configs from separate folder to experiment folders
    configs = {}
    for config in glob.glob(f'{research_name}/configs/*'):
        for experiment_folder in glob.glob(f'{research_name}/results/{glob.escape(os.path.basename(config))}/*'):
            exp_id = os.path.basename(experiment_folder)
            configs[exp_id] = os.path.basename(config)

    for exp_id, config in configs.items():
        src = f'{research_name}/configs/{config}'
        dst = f'{research_name}/results/{config}/{exp_id}/config.dill'
        with open(src, 'rb') as f:
            content = dill.load(f) # content is a ConfigAlias instance
            content['updates'] = content['update'] # Rename column for the new format
            content.pop_config('update')
            content['device'] = None # Add column
        with open(dst, 'wb') as f:
            dill.dump(content, f)
        with open(f'{research_name}/results/{config}/{exp_id}/config.json', 'w') as f:
            json.dump(jsonify(content.config().config), f)

    # Remove folder with configs
    shutil.rmtree(f'{research_name}/configs')

    # Remove one nested level
    initial_results = glob.glob(f'{research_name}/results/*')
    for exp_path in initial_results:
        for path in os.listdir(exp_path):
            src = os.path.join(exp_path, path)
            dst = os.path.join(os.path.dirname(exp_path), path)
            shutil.move(src, dst)
    for path in initial_results:
        shutil.rmtree(path)

    # Rename 'results' folder to 'experiments'
    shutil.move(f'{research_name}/results', f'{research_name}/experiments')

    # Move files from experiment folder to subfodlers
    for results_file in tqdm_notebook(glob.glob(f'{research_name}/experiments/*/*'), disable=(not bar)):
        filename = os.path.basename(results_file)
        content = get_content(results_file)
        if content is not None:
            content.pop('sample_index')
            iterations = content.pop('iteration')

            unit_name, iteration_in_name = filename.split('_')
            iteration_in_name = int(iteration_in_name) - 1
            dirname = os.path.dirname(results_file)
            for var in content:
                new_dict = OrderedDict()
                for i, val in zip(iterations, content[var]):
                    new_dict[i] = val
                folder_for_var = f'{dirname}/results/{unit_name}_{var}'
                if not os.path.exists(folder_for_var):
                    os.makedirs(folder_for_var)
                dst = f'{folder_for_var}/{iteration_in_name}'
                with open(dst, 'wb') as f:
                    dill.dump(new_dict, f)
            os.remove(results_file)

def get_content(path):
    """ Open research results file (if it is research results file, otherwise None). """
    filename = os.path.basename(path)
    if len(filename.split('_')) != 2:
        return None
    _, iteration_in_name = filename.split('_')
    if not iteration_in_name.isdigit():
        return None
    try:
        with open(path, 'rb') as f:
            content = dill.load(f)
    except dill.UnpicklingError:
        return None
    if not isinstance(content, dict):
        return None
    if 'sample_index' not in content or 'iteration' not in content:
        return None
    return content

def jsonify(src):
    """ Transform np.arrays to lists to JSON serialize. """
    src = deepcopy(src)
    for key, value in src.items():
        if isinstance(value, np.ndarray):
            src[key] = value.tolist()
    return src

def create_logger(name, path=None, loglevel='info'):
    """ Create logger. """
    loglevel = getattr(logging, loglevel.upper())
    logger = logging.getLogger(name)
    logger.setLevel(loglevel)

    if path is not None:
        handler = logging.FileHandler(path)
    else:
        handler = logging.StreamHandler() #TODO: filter outputs
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                                datefmt='%y-%m-%d %H:%M:%S')
    handler.setLevel(loglevel)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger

def must_execute(iteration, when, n_iters=None, last=False):
    """ Returns does unit must be executed for the current iteration. """
    if last and 'last' in when:
        return True

    frequencies = (item for item in when if isinstance(item, int) and item > 0)
    iterations = (int(item[1:]) for item in when if isinstance(item, str) and item != 'last')

    it_ok = iteration in iterations
    freq_ok = any((iteration+1) % item == 0 for item in frequencies)

    if n_iters is None:
        return it_ok or freq_ok

    return (iteration + 1 == n_iters and 'last' in when) or it_ok or freq_ok

def parse_name(name):
    """ Parse name of the form 'namespace_name.unit_name' into tuple ('namespace_name', 'unit_name'). """
    if '.' not in name:
        raise ValueError('`func` parameter must be provided or name must be "namespace_name.unit_name"')
    name_components = name.split('.')
    if len(name_components) > 2:
        raise ValueError(f'name must be "namespace_name.unit_name" but {name} were given')
    return name_components

def generate_id(config, random, create_prefix):
    """ Generate id for experiment. """
    name = config.alias()['_prefix'] if create_prefix else ''
    name += hashlib.md5(config.alias(as_string=True).encode('utf-8')).hexdigest()[:8]
    name += ''.join(str(i) for i in random.integers(10, size=8))
    return name

def plot_results_by_config(results, variables, figsize=None, layout=None, **kwargs):
    """
    Given results from Research.run() draws plots of specified variables for all configs

    Parameters
    ----------
    results : pandas.DataFrame
        results produced by Research.run()
    variables : tuple or list
        variables to plot
    figsize : tuple or None
        figsize to pass to matplotlib. If None (default value) figsize is set to (x, y),
        where x = (5 * number of variables), y = (5 * number of configs in `results`)
    layout: 'flat', 'square' or None
        plot arranging strategy when only one variable is needed (default: None, plots are arranged vertically)
    """
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
        for y, val in enumerate(variables):
            ax = axs[n_vars * x + y]

            cols = ['repetition', 'cv_split'] if 'cv_split' in df.columns else 'repetition'

            res = (df.pivot_table(index='iteration', columns=cols, values=val)
                     .rename(columns=lambda s: 'rep ' + str(s), level=0))

            if 'cv_split' in df.columns:
                res = res.rename(columns=lambda s: 'split ' + str(s), level=1)

            res.plot(ax=ax, **kwargs)
            ax.set_title(config)
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
    fontsize = kwargs.pop('fontsize', 28)
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
            ax[i].title.set_size(fontsize)
        ax[i].grid(b=None)

    for i in range(n_items, nrows * ncols):
        fig.delaxes(ax[i])
