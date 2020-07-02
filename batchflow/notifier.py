""" Progress notyfier. """
import math
from time import time, gmtime, strftime

from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook
from tqdm.auto import tqdm as tqdm_auto

import numpy as np
from IPython import display
import matplotlib.pyplot as plt

from .monitor import ResourceMonitor, MONITOR_ALIASES
from .named_expr import NamedExpression, eval_expr


class DummyBar:
    """ Progress tracker without visual representation. """
    #pylint: disable=invalid-name
    def __init__(self, total, *args, **kwargs):
        self.total = total
        self.args, self.kwargs = args, kwargs

        self.n = 0
        self.start_t = time()

    def update(self, n):
        self.n += n

    def format_meter(self, n, total, t, **kwargs):
        _ = kwargs
        return f'{n}/{total} iterations done; elapsed time is {t:3.3} seconds'

    def sp(self, *args, **kwargs):
        _ = args, kwargs

    def set_description(self, *args, **kwargs):
        _ = args, kwargs

    def close(self):
        pass



class Notifier:
    """ Progress tracker and a resource monitor tool in one.

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
    total, batch_size, n_iters, n_epochs, length : int
        Parameters to calculate total amount of iterations.
    drop_last : bool
        Whether the last batch of data is dropped from iterations.
    variables : str, :class:`.NamedExpression` or sequence of them
        Allows to set trackable entities from the pipeline the Notifier is used in:
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
    def __init__(self, bar=None, *args,
                 total=None, batch_size=None, n_iters=None, n_epochs=None, drop_last=False, length=None,
                 variables=None, monitors=None, monitor_kwargs=None, file=None,
                 plot=False, window=None, layout='h', figsize=None, **kwargs):
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

        # Prepare file log
        self.file = file
        if self.file:
            with open(self.file, 'w') as f:
                timestamp = f'{strftime("%Y-%m-%d  %H:%M:%S", gmtime())}'
                msg = 'Notifier started'
                print(f'{timestamp}     {msg}', file=f)


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
        self.timestamps = []
        self.start_monitors()

        # Create bar; set the number of total iterations, if possible
        self.bar = None

        if callable(bar):
            bar_func = bar
        elif bar in ['n', 'nb', 'notebook']:
            bar_func = tqdm_notebook
        elif bar in ['a', 'auto']:
            bar_func = tqdm_auto
        elif bar in ['j', 'jpn', 'jupyter']:
            bar_func = tqdm_notebook
            plot = True
        elif bar in [True, 't', 'tqdm']:
            bar_func = tqdm
        else:
            bar_func = DummyBar
        self.bar_func = lambda total: bar_func(total=total, *args, **kwargs)
        self.update_total(total=total, batch_size=batch_size, n_iters=n_iters, n_epochs=n_epochs,
                          drop_last=drop_last, length=length)

        # Prepare plot params
        #pylint: disable=invalid-unary-operand-type
        self.plot = plot
        self.slice = slice(-window, None, None) if isinstance(window, int) else slice(None)
        self.layout = (1, len(self.data_generators)) if layout.startswith('h') else (len(names), 1)
        self.figsize = figsize or ((20, 5) if layout.startswith('h') else (20, 5*(len(names))))


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
        """ Update Notifier with new info:
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

        if self.file:
            self.update_file()

        if self.has_monitors and self.bar.n < self.bar.total:
            self.start_monitors()

    def update_data(self, pipeline=None, batch=None):
        """ Get data from monitor or pipeline. """
        self.timestamps.append(gmtime())

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

        description = '   '.join(description)
        self.bar.set_description(description)

    def update_plots(self, add_suptitle=False, savepath=None):
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
            title = self.format_meter(self.n, self.total, time()-self.start_t, ncols=80)
            plt.suptitle(title, y=0.99, fontsize=14)

        if savepath:
            plt.savefig(savepath, bbox_inches='tight', pad_inches=0)
        plt.show()

    # Convenient alias for working inspect instance after pipeline run
    visualize = update_plots

    def update_file(self):
        """ Update file on the fly. """
        with open(self.file, 'a+') as f:
            timestamp = strftime("%Y-%m-%d  %H:%M:%S", self.timestamps[-1])
            msg = f'Iteration {self.bar.n:5};    {self.bar.desc[:-1]}'
            print(f'{timestamp}     {msg}', file=f)

    def to_file(self, file):
        """ Log all the iteration-wise info (timestamps, descriptions) into file."""
        with open(file, 'w') as f:
            for i, timestamp in enumerate(self.timestamps):
                timestamp_ = strftime("%Y-%m-%d  %H:%M:%S", timestamp)

                description = []
                for name, generator in self.data_generators.items():
                    if not isinstance(generator, ResourceMonitor):
                        value = self.data[name][i]
                        desc = f'{name}={value:6.6}' if isinstance(value, float) else f'{name}={value:6}'
                        description.append(desc)
                description = '   '.join(description)

                msg = f'Iteration {i:5};    {description}'
                print(f'{timestamp_}     {msg}', file=f)


    def __call__(self, iterable):
        self.update_total(0, 0, 0, 0, 0, total=len(iterable))
        for item in iterable:
            yield item
            self.update()
        self.close()


    def close(self):
        """ Close the underlying progress bar. """
        self.bar.close()
