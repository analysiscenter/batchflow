""" Progress notifier. """
import math
from time import time, gmtime, strftime

from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook
from tqdm.auto import tqdm as tqdm_auto

import numpy as np
import matplotlib.pyplot as plt
try:
    from IPython import display
except ImportError:
    pass

from .monitor import ResourceMonitor, MONITOR_ALIASES
from .named_expr import NamedExpression, eval_expr


class DummyBar:
    """ Progress tracker without visual representation. """
    #pylint: disable=invalid-name
    def __init__(self, total, *args, **kwargs):
        self.total = total
        self.args, self.kwargs = args, kwargs

        self.n = 0
        self.desc = ''
        self.start_t = time()

    def update(self, n):
        self.n += n

    def format_meter(self, n, total, t, **kwargs):
        _ = kwargs
        return f'{n}/{total} iterations done; elapsed time is {t:3.3} seconds'

    def sp(self, *args, **kwargs):
        _ = args, kwargs

    def set_description(self, description):
        self.desc = description

    def close(self):
        pass



class Notifier:
    """ Progress tracker and a resource monitor tool in one.
    Allows to dynamically track and display containers (pipeline variables, images, monitor),
    log them to file in both textual and visual formats.

    Instance can be used to wrap iterators or by calling :meth:`.update` manually.

    Parameters
    ----------
    bar : {'n', 'a', 'j', True} or callable
        Sets the type of used progress bar:
            - `callable` must provide a tqdm-like interface.
            - `n` stands for notebook version of tqdm bar.
            - `a` stands for automatic choise of appropriate tqdm bar.
            - `j` stands for graph drawing as a progress bar.
            - `t` or True for standard text tqdm is used.
            - otherwise, no progress bar will be displayed. Note that iterations,
            as well as everything else (monitors, variables, logs) are still tracked.
    update_total : bool
        Whether the total amount of iterations should be computed at initialization.
    total, batch_size, n_iters, n_epochs, drop_last, length
        Parameters to calculate total amount of iterations.
    frequency : int
        Frequency of notifier updates.
    monitors : str, :class:`.Monitor`, :class:`.NamedExpression`, dict or sequence of them
        Set tracked ('monitored') entities: they are displayed in the bar description.
        Strings are either registered monitor identifiers or names of pipeline variables.
        Named expressions are evaluated with the pipeline.
        If dict, then 'source' key should be one of the above to identify container.
        Other available keys:
            - 'name' is used to display at bar descriptions and plot titles
            - 'plot_function' is used to display container data.
            Can be used to change the default way of displaying graphs.
    graphs : str, :class:`.Monitor`, :class:`.NamedExpression`, or sequence of them
        Same semantics, as `monitors`, but tracked entities are displayed in dynamically updated plots.
    file : str
        If provided, a textual log is written into the supplied path.
    window : int
        Allows to plot only the last `window` values from every tracked container.
    layout : str
        If `h`, then subplots are drawn horizontally; vertically otherwise.
    figsize : tuple of numbers
        Total size of drawn figure.
    savepath : str
        Path to save image, created by tracking entities with `graphs`.
    *args, **kwargs
        Positional and keyword arguments that are used to create underlying progress bar.
    """
    def __init__(self, bar=None, *args, update_total=True,
                 total=None, batch_size=None, n_iters=None, n_epochs=None, drop_last=False, length=None,
                 frequency=1, monitors=None, graphs=None, file=None,
                 window=None, layout='h', figsize=None, savepath=None, **kwargs):

        # Prepare data containers like monitors and pipeline variables
        if monitors:
            monitors = monitors if isinstance(monitors, (tuple, list)) else [monitors]
        else:
            monitors = []

        if graphs:
            graphs = graphs if isinstance(graphs, (tuple, list)) else [graphs]
        else:
            graphs = []

        self.has_monitors = False
        self.has_graphs = len(graphs) > 0
        self.n_monitors = len(monitors)

        self.data_containers = []
        for container in monitors + graphs:
            if not isinstance(container, dict):
                container = {'source': container}

            if isinstance(container['source'], str) and container['source'].lower() in MONITOR_ALIASES:
                container['source'] = MONITOR_ALIASES[container['source'].lower()]()

            source = container.get('source')
            if source is None:
                raise ValueError('Passed dictionaries as `monitors` or `graphs` should contain `source` key!')

            if isinstance(source, ResourceMonitor):
                self.has_monitors = True

            if 'name' not in container:
                if isinstance(source, ResourceMonitor):
                    container['name'] = source.__class__.__name__
                elif isinstance(source, NamedExpression):
                    container['name'] = source.name
                elif isinstance(source, str):
                    container['name'] = source

            self.data_containers.append(container)

        self.frequency = frequency
        self.timestamps = []
        self.start_monitors()

        # Prepare file log
        self.file = file
        if self.file:
            with open(self.file, 'w') as _:
                pass

        # Create bar; set the number of total iterations, if possible
        self.bar = None

        bar_func = None
        if callable(bar):
            bar_func = bar
        elif bar in ['n', 'nb', 'notebook', 'j', 'jpn', 'jupyter']:
            bar_func = tqdm_notebook
        elif bar in ['a', 'auto']:
            bar_func = tqdm_auto
        elif bar in [True, 't', 'tqdm']:
            bar_func = tqdm
        elif bar in [False, None]:
            bar_func = DummyBar
        else:
            raise ValueError('Unknown bar value:', bar)

        # Set default values for bars
        if 'ncols' not in kwargs:
            if bar_func == tqdm_notebook:
                kwargs['ncols'] = min(700 + 100 * len(monitors or []), 1000)
            elif bar_func == tqdm:
                kwargs['ncols'] = min(80 + 10 * len(monitors or []), 120)

        self.bar_func = lambda total: bar_func(total=total, *args, **kwargs)

        if update_total:
            self.update_total(total=total, batch_size=batch_size, n_iters=n_iters, n_epochs=n_epochs,
                              drop_last=drop_last, length=length)

        # Prepare plot params
        #pylint: disable=invalid-unary-operand-type
        self.slice = slice(-window, None, None) if isinstance(window, int) else slice(None)
        self.layout, self.figsize, self.savepath = layout, figsize, savepath


    def update_total(self, batch_size, n_iters, n_epochs, drop_last, length, total=None):
        """ Re-calculate total number of iterations. """
        if total is None:
            if n_iters is not None:
                total = n_iters
            if n_epochs is not None:
                if drop_last:
                    total = length // batch_size * n_epochs
                else:
                    total = math.ceil(length * n_epochs / batch_size)

        # Force close previous bar, create new
        if self.bar is not None:
            try:
                # jupyter bar must be closed and reopened
                self.bar.sp(close=True)
                self.bar = self.bar_func(total=total)
            except TypeError:
                # text bar can work with a simple reassigning of `total`
                self.bar.total = total
        else:
            self.bar = self.bar_func(total=total)


    def update(self, n=1, pipeline=None, batch=None):
        """ Update Notifier with new info:
        - fetch up-to-date data from batch, pipeline and monitors
        - set bar description
        - draw plots anew
        - update log file
        - increment underlying progress bar tracker
        """
        if (self.bar.n + 1) % self.frequency == 0 or (self.bar.n == self.bar.total - 1):
            self.timestamps.append(gmtime())

            if self.data_containers:
                self.update_data(pipeline=pipeline, batch=batch)
                self.update_description()

            if self.has_graphs:
                self.update_plots(self.n_monitors, True)

            if self.file:
                self.update_file()

        self.bar.update(n)

    def update_data(self, pipeline=None, batch=None):
        """ Get data from monitor or pipeline. """
        for container in self.data_containers:
            source = container['source']
            if isinstance(source, ResourceMonitor):
                source.fetch()
                container['data'] = source.data

            elif isinstance(source, NamedExpression):
                value = eval_expr(source, pipeline=pipeline, batch=batch)
                container['data'] = value

            elif isinstance(source, str):
                value = pipeline.v(source)
                container['data'] = value

    def update_description(self):
        """ Set new bar description. """
        description = self.create_description(iteration=-1)
        self.bar.set_description(description)

    def update_plots(self, index=0, add_suptitle=False, savepath=None, clear_display=True):
        """ Draw plots anew. """
        num_graphs = len(self.data_containers) - index
        layout = (1, num_graphs) if self.layout.startswith('h') else (num_graphs, 1)
        figsize = self.figsize or ((20, 5) if self.layout.startswith('h') else (20, 5*num_graphs))

        if clear_display:
            display.clear_output(wait=True)
        fig, ax = plt.subplots(*layout, figsize=figsize)
        ax = ax if isinstance(ax, np.ndarray) else [ax]

        for i, container in enumerate(self.data_containers):
            if i >= index:
                source = container['source']
                name = container['name']
                plot_function = container.get('plot_function')

                if isinstance(source, ResourceMonitor):
                    data_x = np.array(source.ticks)[self.slice] - source.ticks[0]
                    data_y = source.data[self.slice]
                    x_label, y_label = 'Time, s', source.UNIT
                else:
                    data_y = container['data']
                    data_x = list(range(len(data_y)))[self.slice]
                    data_y = data_y[self.slice]
                    x_label, y_label = 'Iteration', ''

                if plot_function is not None:
                    plot_function(fig=fig, ax=ax[i - index], i=i,
                                  data_x=data_x, data_y=data_y, container=container)
                # Default plotting functionality
                elif isinstance(data_y, (tuple, list)) or (isinstance(data_y, np.ndarray) and data_y.ndim == 1):
                    ax[i - index].plot(data_x, data_y)
                    ax[i - index].set_title(name, fontsize=12)
                    ax[i - index].set_xlabel(x_label, fontsize=12)
                    ax[i - index].set_ylabel(y_label, fontsize=12, rotation='horizontal', labelpad=15)
                    ax[i - index].grid(True)
                elif isinstance(data_y, np.ndarray) and data_y.ndim == 2:
                    ax[i - index].imshow(data_y)
                    ax[i - index].set_title(name, fontsize=12)

        if add_suptitle:
            title = self.format_meter(self.n+1, self.total, time()-self.start_t, ncols=80)
            plt.suptitle(title, y=0.99, fontsize=14)

        savepath = savepath or (f'{self.savepath}_{self.bar.n}' if self.savepath is not None else None)
        if savepath:
            plt.savefig(savepath, bbox_inches='tight', pad_inches=0)
        plt.show()

    def update_file(self):
        """ Update file on the fly. """
        with open(self.file, 'a+') as f:
            print(self.create_message(self.bar.n, self.bar.desc[:-2]), file=f)


    def visualize(self):
        """ Convenient alias for working with an instance. """
        self.update_plots(clear_display=False)

    def to_file(self, file):
        """ Log all the iteration-wise info (timestamps, descriptions) into file."""
        with open(file, 'w') as f:
            for i in range(self.bar.n):
                description = self.create_description(iteration=i)
                print(self.create_message(i, description), file=f)

    def __call__(self, iterable):
        self.update_total(0, 0, 0, 0, 0, total=len(iterable))
        for item in iterable:
            yield item
            self.update()
        self.close()

    def close(self):
        """ Close the underlying progress bar. """
        self.bar.close()
        self.stop_monitors()


    # Utility functions
    def start_monitors(self):
        """ Start collection of data for every resource monitor. """
        for container in self.data_containers:
            source = container['source']
            if isinstance(source, ResourceMonitor):
                source.start()

    def stop_monitors(self):
        """ Stop collection of data for every resource monitor. """
        for container in self.data_containers:
            source = container['source']
            if isinstance(source, ResourceMonitor):
                source.stop()

    def create_description(self, iteration):
        """ Create string description of a given iteration. """
        description = []
        for container in self.data_containers:
            source = container['source']
            name = container['name']
            if isinstance(source, (str, NamedExpression)):
                value = container['data'][iteration]
                if isinstance(value, (int, float, np.signedinteger, np.floating)):
                    desc = f'{name}={value:<6.6f}' if isinstance(value, (float, np.floating)) else f'{name}={value:<6}'
                    description.append(desc)
        return ';   '.join(description)

    def create_message(self, iteration, description):
        """ Combine timestamp, iteration and description into one string message. """
        timestamp = strftime("%Y-%m-%d  %H:%M:%S", self.timestamps[iteration])
        return f'{timestamp}     Iteration {iteration:5};    {description}'

    def __getattr__(self, key):
        """ Redirect everything to the underlying bar. """
        if not key in self.__dict__ and hasattr(self.bar, key):
            return getattr(self.bar, key)
        raise AttributeError(key)
