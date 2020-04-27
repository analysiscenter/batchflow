""" Progress notyfier. """
import sys
import math
from time import time

from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook
from tqdm.auto import tqdm as tqdm_auto

import numpy as np
from IPython import display
import matplotlib.pyplot as plt

from .monitor import ResourceMonitor, MONITOR_ALIASES
from .named_expr import NamedExpression, eval_expr

class Notifyer:
    """ Progress tracker and a resource monitor tool in one.

    Parameters
    ----------
    bar : {'n', 'a', 'j', True} or callable
        Sets the type of used progress bar:
            - `callable` must provide a tqdm-like interface.
            - `n` stands for notebook version of tqdm bar.
            - `a` stands for automatic choise of appropriate tqdm bar.
            - `j` stands for graph drawing as a progress bar.
            - otherwise, standart tqdm is used.
    total, batch_size, n_iters, n_epochs, length : int
        Parameters to calculate total amount of iterations.
    drop_last : bool
        Whether the last batch of data is dropped from iterations.
    variables : str, :class:`.NamedExpression` or sequence of them
        Allows to set trackable entities from the pipeline the Notifyer is used in:
        If str, then stands for name of the variable to get from the pipeline.
        If any of the named expressions, then evaluated with the pipeline.
    monitors : str, :class:`.Monitor` or sequence of them
        Allows to monitor resources. Strings should be registered aliases for monitors like `cpu`, `gpu`, etc.
    monitor_kwargs : dict
        Parameters of monitor creation like `frequency`, `pid`, etc.
    plot : bool
        If True, then tracked data (usually list of values like memory usage or loss over training process)
        is dynamically tracked on graphs. Note that rendering takes a lot of time.
    window : int
        Allows to plot only the last `window` values from every tracked container.
    layout : str
        If `h`, then subplots are drawn horizontally; vertically otherwise.
    figsize : tuple of numbers
        Total size of drawn figure.
    *args, **kwargs
        Positional and keyword arguments that are used to create underlying progress bar.
    """
    def __init__(self, bar=None, total=None, *args,
                 batch_size=None, n_iters=None, n_epochs=None, drop_last=False, length=None,
                 variables=None, monitors=None, monitor_kwargs=None,
                 plot=False, window=None, layout='h', figsize=None, **kwargs):

        # Create bar; set number of total iterations, if possible
        if callable(bar):
            bar_func = bar
        elif bar == 'n':
            bar_func = tqdm_notebook
        elif bar == 'a':
            bar_func = tqdm_auto
        elif bar == 'j':
            bar_func = tqdm_notebook
            plot = True
        else:
            bar_func = tqdm
        self.bar = bar_func(*args, **kwargs)
        self.update_total(total=total, batch_size=batch_size, n_iters=n_iters, n_epochs=n_epochs,
                          drop_last=drop_last, length=length)

        data_generators = []

        # Prepare variables
        if variables is not None:
            variables = variables if isinstance(variables, (tuple, list)) else [variables]
            data_generators.extend(variables)

        # Prepare monitors
        if monitors is not None:
            monitors = 'MemoryMonitor' if monitors is True else monitors
            monitors = [monitors] if not isinstance(monitors, (tuple, list)) else monitors
            monitors = [MONITOR_ALIASES[name.lower()](**(monitor_kwargs or {})) if isinstance(name, str) else name
                        for name in monitors]
            data_generators.extend(monitors)
            self.has_monitors = True
        else:
            self.has_monitors = False

        # Prepare containers for data
        names = []
        for generator in data_generators:
            if isinstance(generator, ResourceMonitor):
                names.append(generator.__class__.__name__)
            elif isinstance(generator, NamedExpression):
                names.append(generator.name)
            elif isinstance(generator, str):
                names.append(generator)

        self.data = {name: [] for name in names}
        self.data_generators = dict(zip(names, data_generators))
        self.start_monitors()

        # Prepare plot params
        self.plot = plot
        self.slice = slice(None) if window is None else slice(-window, None, None)
        self.layout = (1, len(names)) if layout.startswith('h') else (len(names), 1)
        self.figsize = figsize or ((20, 5) if layout.startswith('h') else (20, 5*(len(names))))


    def update_total(self, batch_size, n_iters, n_epochs, drop_last, length, total=None):
        """ Re-calculate total number of iterations. """
        if total is None:
            if n_iters is not None:
                total = n_iters
            elif n_epochs is None:
                total = sys.maxsize
            elif drop_last:
                total = length // batch_size * n_epochs
            else:
                total = math.ceil(length * n_epochs / batch_size)
        self.bar.total = total

    def __getattr__(self, key):
        """ Redirect everything to the underlying bar. """
        if not key in self.__dict__ and hasattr(self.bar, key):
            return getattr(self.bar, key)
        raise AttributeError(key)


    def start_monitors(self):
        """ Start collection of data for every resource monitor. """
        for monitor in self.data_generators.values():
            if isinstance(monitor, ResourceMonitor):
                monitor.start()


    def update(self, n=1, pipeline=None, batch=None):
        """ Update Notifyer with new info:
            - increment underlying progress bar tracker
            - set bar description
            - fetch up-to-date data from pipeline and batch; gather info from monitors
            - draw plots anew
            - re-start monitors
        """
        self.bar.update(n)

        self.update_data(pipeline=pipeline, batch=batch)
        self.update_description()

        if self.plot:
            self.update_plots(True)

        if self.has_monitors and self.bar.n < self.bar.total:
            self.start_monitors()

    def update_data(self, pipeline=None, batch=None):
        """ Get data from monitor or pipeline. """
        for name, generator in self.data_generators.items():

            if isinstance(generator, ResourceMonitor):
                value = generator.stop()
                self.data[name].extend(value)

            elif isinstance(generator, NamedExpression):
                value = eval_expr(generator, pipeline=pipeline, batch=batch)
                self.data[name].append(value)

            elif isinstance(generator, str):
                value = pipeline.v(generator)
                self.data[name].append(value)

    def update_description(self):
        """ Set new bar description. """
        description = []
        for name, generator in self.data_generators.items():
            if not isinstance(generator, ResourceMonitor):
                value = self.data[name][-1]
                desc = f'{name}={value:5.5}' if isinstance(value, float) else f'{name}={value:5}'
                description.append(desc)

        description = '>>>'.join(description)
        self.bar.set_description(description)

    def update_plots(self, add_suptitle=False):
        """ Draw plots anew. """
        display.clear_output(wait=True)
        _, ax = plt.subplots(*self.layout, figsize=self.figsize)
        ax = ax if isinstance(ax, np.ndarray) else [ax]

        for i, (name, generator) in  enumerate(self.data_generators.items()):
            if isinstance(generator, ResourceMonitor):
                data_x = np.array(generator.ticks)[self.slice] - generator.ticks[0]
                data_y = generator.data[self.slice]
            else:
                data_y = self.data[name]
                data_x = list(range(len(data_y)))[self.slice]
                data_y = data_y[self.slice]

            ax[i].plot(data_x, data_y)
            ax[i].set_title(name, fontsize=12)
            ax[i].grid(True)

        if add_suptitle:
            plt.suptitle(self.format_meter(self.n, self.total, time()-self.start_t, ncols=80), y=0.99, fontsize=14)
        plt.show()

    # Convenient alias for working inspect instance after pipeline run
    visualize = update_plots


    def close(self):
        """ Close the underlying progress bar. """
        self.bar.close()
