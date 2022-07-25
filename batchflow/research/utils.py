""" Auxilary functions """
import os
import logging
import hashlib
import io
import contextlib
import warnings
from copy import deepcopy
import dill
import numpy as np
import pandas as pd

from ..plotter import plot
from ..plotter.plot import Subplot
from ..utils import to_list


class MultiOut:
    """ Wrapper for several outputs streams. """
    def __init__(self, *args):
        self.handlers = args

    def write(self, s):
        for f in self.handlers:
            f.write(s + '\n' + '-' * 30 + '\n')

    def flush(self):
        for f in self.handlers:
            f.flush()

    def __getattr__(self, attr):
        return getattr(self.handlers[0], attr)

class Unpickler(dill.Unpickler):
    """ Unpickler which will load object as a string if it can't be found. Is necessary
    to deal with objects imported from modules and removed. """
    def find_class(self, module, name):
        """ Get object class. """
        try:
            return super().find_class(module, name)
        except AttributeError:
            warnings.warn(f"Can't get attribute {name} on <module {module}>")
            return f"<object {module}.{name}>"

def deserialize(file, ignore=None, **kwargs):
    """ Unpickle an object from a file. Attributed that can't be loaded will be changed by str. """
    return Unpickler(file, ignore=ignore, **kwargs).load()

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

def generate_id(config, random, create_prefix=False):
    """ Generate id for experiment. """
    name = config.alias()['_prefix'] if create_prefix else ''
    name += hashlib.md5(config.alias(as_string=True).encode('utf-8')).hexdigest()[:8]
    name += ''.join(str(i) for i in random.integers(10, size=8))
    return name

def create_output_stream(redirect, dump=False, filename=None, path=None, common=True):
    """ Create stream to redirect stdout/stderr. """
    if bool(redirect):
        values = [1, 3] if common else [2, 3]
        if redirect in values:
            if dump:
                filename = os.path.join(path, filename)
                file = open(filename, 'a')
            else:
                file = io.StringIO()
        else:
            file = open(os.devnull, 'w')
    else:
        file = contextlib.nullcontext()
    return file

def plot_research(df, variables=None, subplots=None, aggregate=None, aggregate_fn='mean', same_color=None, layout=None,
                  short_label=True, ignore=('id', 'name'), meta=('id', 'config', 'repetition', 'cv_split'), **kwargs):
    """ Plot graphs of variables logged during research. Subplots are grouped by variables name by default.
    All dataframe columns are treated as variables except 'iteration' and also those provided in `ignore` and `meta`.


    Parameters
    ----------
    df : pd.DataFrame
        Dataframe of results produced by `Research.run`.
    variables : None, str or list of str
        Name(s) of dataframe column(s) containing variable(s) values to plot.
        By default all dataframe columns are treated as `variables` except those from `meta`.
    subplots : None, str or list of str
        Name(s) of dataframe column(s) to use for grouping data over subplots.
        By default research results are aggregated over `variables`.
    aggregate : str or list of str
        Name(s) of dataframe columns to use for aggregating on every subplot.
        By default data is not aggregated.
    aggregate_fn : str or callable
        Determines how data is aggregated. Must be a valid `aggfunc` parameter for `pd.pivot_table`.
    same_color : None or str
        Color values of same groups with same colors. By default color every value sequence with its own color.
        E.g. `plot_research(results_df, variables='loss', same_color='config')` will force plotter to use same color for
        loss values corresponding to same config.
    layout : None, 'horizontal' or 'vertical'
        Determines how subplots should be arranged.
        If 'horizontal', subplots are stretched in a single row, if 'vertical' — in a single column.
        By default number of plot rows is set to the number of displayed variables
        and number of columns is calculated to accomodate the plots.
    short_label : bool
        Whether shorten legend labels, omitting layer groupping parameters names and keeping only their values.
        E.g. "{'optimizer': 'Adam'}, 0" instead of "config: {'optimizer': 'Adam'}, repetition: 0".
    ignore : None, str or list of str
        Name(s) of column(s) to ignore.
    meta : str or list of str
        Name(s) of column(s) to use for grouping data over subplots/layers.
        Has lower priority than `ignore`, so if same name provided in both arguments, it's ignored.
    """
    meta = to_list(meta)
    ignore = to_list(ignore)

    if variables is None:
        variables = sorted(set(df.columns).difference(['iteration', *meta, *ignore]))
    variables = to_list(variables)

    if subplots is None:
        subplots = []
    subplots = to_list(subplots)

    if aggregate is None:
        aggregate = []
    elif aggregate is True:
        aggregate = ['repetition']
    else:
        aggregate = to_list(aggregate)

    for name in aggregate:
        if name not in df.columns:
            raise ValueError(f"Cannot aggregate data over `{name}` — nonexisting column name provided.")

    meta = [name for name in meta if name not in aggregate and name not in ignore and name in df.columns]
    subplots = [name for name in subplots if name in meta]
    layers = [name for name in meta if name not in subplots]

    if same_color is None:
        if len(layers) == 0:
            unique_layers = [None]
        else:
            unique_layers = df[layers].drop_duplicates().apply(lambda row: tuple(row), axis=1).values
    else:
        if same_color not in layers:
            msg = f"Can't color lines with same `{same_color}`"
            if same_color in subplots:
                msg += " since data is already grouped by this column over subplots."
            elif same_color in aggregate:
                msg += " since data is already aggregated over this column."
            else:
                msg += " since column with such name is not present in provided dataframe."
            raise ValueError(msg)
        unique_layers = df[same_color].unique()
    name_to_color = {value: Subplot.CURVE_COLORS[index] for index, value in enumerate(unique_layers)}

    data = []
    color = []
    title = []
    label = []
    ylabel = []

    pivot_df = df.pivot_table(index='iteration', columns=subplots + layers, values=variables, aggfunc=aggregate_fn)
    if len(layers) == 0:
        index = pivot_df.columns
    else:
        index = set(column[:len(subplots) + 1] for column in pivot_df.columns)

    for subplot_index in sorted(index):
        subplot_data = []
        subplot_color = []
        subplot_label = []

        subplot_df = pivot_df[subplot_index]

        if isinstance(subplot_df, pd.Series):
            subplot_df = subplot_df.to_frame()

        for layer_index in subplot_df:
            layer_df = subplot_df[layer_index]

            layer_data = (layer_df.index.values, layer_df.values)
            subplot_data.append(layer_data)

            layer_index = layer_index if isinstance(layer_index, tuple) else (layer_index, )

            if same_color is None:
                layer_id = layer_index if len(layers) > 0 else None
            else:
                layer_id = layer_index[layers.index(same_color)]
            layer_color = name_to_color[layer_id]
            subplot_color.append(layer_color)

            if len(layers) == 0:
                layer_label = None if len(aggregate) == 0 else f"aggregated by {aggregate}"
            elif short_label:
                layer_label = ', '.join(map(str, layer_index))
            else:
                layer_label = ', '.join([': '.join(map(str, x)) for x in zip(layers, layer_index)])
            subplot_label.append(layer_label)

        data.append(subplot_data)
        if same_color is not None:
            color.append(subplot_color)

        subplot_title = ', '.join(
            [
                ': '.join(map(str, x)) for x in
                zip(
                    ['variable', *subplots],
                    to_list(subplot_index)
                )
            ]
        )
        title.append(subplot_title)
        label.append(subplot_label)
        subplot_ylabel = str(subplot_index[0]).replace('_', ' ').capitalize()
        ylabel.append(subplot_ylabel)

    plot_config = {
        'subplot_width': 8,
        'title': title,
        'label': label,
        'xlabel': 'Iteration',
        'ylabel': ylabel,
        **kwargs
    }

    if 'ncols' not in kwargs and 'nrows' not in kwargs:
        if layout is None:
            if len(index) == len(variables):
                ncols, nrows = len(variables), 1
            else:
                ncols, nrows = len(index) // len(variables), len(variables)
        elif layout == 'vertical':
            ncols, nrows = 1, len(index)
        elif layout == 'horizontal':
            ncols, nrows = len(index), 1

        plot_config = {**plot_config, 'ncols': ncols, 'nrows': nrows, 'ratio': nrows / ncols}

    if same_color is not None:
        plot_config = {'color': color, **plot_config}

    return plot(data, mode='curve', **plot_config)

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
        repetition_cols = [f'　(repetition {i})' for i in ndf['repetition'].unique()]
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
