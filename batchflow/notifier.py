""" Progress notifier. """
import sys
import math
import warnings
from time import time, gmtime, strftime

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from tqdm import tqdm
    from tqdm.notebook import tqdm as tqdm_notebook
    from tqdm.autonotebook import tqdm as tqdm_auto

import numpy as np

try:
    from IPython import display
except ImportError:
    pass

from .named_expr import NamedExpression, eval_expr
from .monitor import ResourceMonitor, MONITOR_ALIASES
from .utils_telegram import TelegramMessage
from .plotter import plot
from .plotter.utils import PlotConfig



class DummyBar:
    """ Progress tracker without visual representation. """
    #pylint: disable=invalid-name
    def __init__(self, total, *args, **kwargs):
        self.total = total
        self.args, self.kwargs = args, kwargs

        self.n = 0
        self.desc = ''
        self.postfix = ''
        self.start_t = time()

    def update(self, n):
        self.n += n

    @property
    def format_dict(self):
        return {'n': self.n, 'total': self.total, 't': time() - self.start_t}

    def format_meter(self, n, total, t, **kwargs):
        _ = kwargs
        return f'{n}/{total} iterations done; elapsed time is {t:3.3} seconds; {self.postfix}'

    def display(self, *args, **kwargs):
        _ = args, kwargs

    def set_description(self, desc):
        self.desc = desc

    def set_postfix_str(self, postfix):
        self.postfix = postfix

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
    desc : str
        Prefix for created descriptions.
    disable : bool
        Whether to disable the notifier completely: progress bar, monitors and graphs.
    total, batch_size, n_iters, n_epochs, drop_last, length
        Parameters to calculate total amount of iterations.

    frequency : int
        Frequency of notifier updates.
    monitors : str, :class:`.Monitor`, :class:`.NamedExpression`, callable, sequence, dict or sequence of them
        Set tracked ('monitored') entities: they are displayed in the bar description.
        If str, then either registered monitor identifiers or names of pipeline variables.
        Named expressions are evaluated with the pipeline.
        If callable, then it is used to retrieve the container with data.
        If sequence, then it is used as the container with data.
        If dict, then 'source' key should be one of the above to identify container.
        Other available keys:
            - 'name' is used to display at bar descriptions and plot titles. Not used if the `format` is provided.
            - 'format' is used to create string description from the last value in the container's data.
            - 'plot_function' is used to display container data.
            Can be used to change the default way of displaying graphs.
    graphs : str, :class:`.Monitor`, :class:`.NamedExpression`, or sequence of them
        Same semantics, as `monitors`, but tracked entities are displayed in dynamically updated plots.
    log_file : str
        If provided, a textual log is written into the supplied path.

    telegram : bool
        Whether to send notifications to a Telegram Bot. Works with both textual bars and figures (from `graphs`).
        Under the hood, keeps track of two messages - one with text, one with media, and edits them when needed.
        `silent` parameters controls, whether messages are sent with notifications or not.

        One must supply telegram `token` and `chat_id` either by passing directly or
        setting environment variables `TELEGRAM_TOKEN` and `TELEGRAM_CHAT_ID`. To get them:
            - create a bot <https://core.telegram.org/bots#6-botfather> and copy its `{token}`
            - add the bot to a chat and send it a message such as `/start`
            - go to <https://api.telegram.org/bot`{token}`/getUpdates> to find out the `{chat_id}`

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
    #pylint: disable=too-many-arguments
    COLOUR_RUNNING = '#2196f3'
    COLOUR_SUCCESS = '#4caf50'
    COLOUR_FAILURE = '#f44336'

    def __init__(self, bar='a', disable=False, frequency=1, monitors=None, graphs=None, log_file=None,
                 total=None, batch_size=None, n_iters=None, n_epochs=None, drop_last=False, length=None,
                 telegram=False, initialize_telegram=False, token=None, chat_id=None, silent=True,
                 window=None, savepath=None, plot_config=None, **kwargs):
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

        plot_config = plot_config or {}
        self.plot_config = plot_config
        if self.has_graphs:
            self.plotter = self.make_plotter(num_graphs=len(graphs), **plot_config)
        else:
            self.plotter = None

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
                    container['name'] = repr(source)
                elif isinstance(source, str):
                    container['name'] = source
                elif callable(source):
                    container['name'] = '<unknown_callable>'
                else:
                    container['name'] = '<unknown_container>'

            self.data_containers.append(container)

        self.frequency = frequency
        self.timestamps = []
        self.start_monitors()

        # Prepare file log
        self.log_file = log_file
        if self.log_file:
            with open(self.log_file, 'w', encoding='utf-8'):
                pass

        # Parse string bar identifier to constructor
        if callable(bar):
            bar_func = bar
        elif bar in ['n', 'nb', 'notebook', 'j', 'jpn', 'jupyter']:
            bar_func = tqdm_notebook
        elif bar in [True, 'a', 'auto']:
            bar_func = tqdm_auto
        elif bar in ['t', 'tqdm']:
            bar_func = tqdm
        elif bar in ['telegram', 'tg']:
            bar_func = tqdm_auto
            telegram = True
        elif bar in [False, None]:
            bar_func = DummyBar
        else:
            raise ValueError('Unknown bar value:', bar)

        # Convert the bar, based on other parameters
        if self.has_graphs and bar_func is tqdm:
            bar_func = tqdm_notebook

        # Set default values for bars
        if bar_func is tqdm or bar_func is tqdm_notebook:
            if bar_func is tqdm:
                ncols = min(80 + 10 * self.n_monitors, 120)
                colour = self.COLOUR_SUCCESS
            elif bar_func is tqdm_notebook:
                ncols = min(700 + 150 * self.n_monitors, 1000)
                colour = None

            kwargs = {
                'ncols': ncols,
                'colour': colour,
                'file': sys.stdout,
                **kwargs
            }

        # Function to initialize / reinitialize bar, when needed
        if disable:
            bar_func = DummyBar
            self.disable()

        self.bar = None
        self.bar_func = lambda total: bar_func(total=total, **kwargs)

        # Make bar with known / unknown total length
        self.total = None
        self.compute_total(total=total, batch_size=batch_size, n_iters=n_iters, n_epochs=n_epochs,
                           drop_last=drop_last, length=length)
        self.make_bar()

        # Prepare plot params
        #pylint: disable=invalid-unary-operand-type
        self.slice = slice(-window, None, None) if isinstance(window, int) else slice(None)
        self.savepath = savepath

        # Prepare Telegram notifications
        self.telegram = telegram
        if self.telegram or initialize_telegram:
            self.telegram_text = TelegramMessage(token=token, chat_id=chat_id, silent=silent)
            self.telegram_media = TelegramMessage(token=token, chat_id=chat_id, silent=silent)


    def compute_total(self, batch_size, n_iters, n_epochs, drop_last, length, total=None):
        """ Re-calculate total number of iterations. """
        if total is None:
            if n_iters is not None:
                total = n_iters
            if n_epochs is not None:
                if drop_last:
                    total = length // batch_size * n_epochs
                else:
                    total = math.ceil(length * n_epochs / batch_size)
        self.total = total

    def make_bar(self):
        """ Create new bar. Force close the previous, if needed. """
        if self.bar is not None:
            try:
                # jupyter bar must be closed and reopened
                self.bar.display(close=True)
                self.bar = self.bar_func(total=self.total)
            except TypeError:
                # text bar can work with a simple reassigning of `total`
                self.bar.total = self.total
        else:
            self.bar = self.bar_func(total=self.total)

    def refresh(self):
        """ Remake the bar, if needed, while keeping the tracked number of passed iterations. """
        if self.total != self.bar.total:
            n = self.bar.n
            self.make_bar()

            if self.bar.n != n:
                self.bar.update(n)

    def disable(self):
        """ Completely disable notifier: progress bar, monitors and graphs. """
        if self.bar is not None:
            try:
                # jupyter bar must be closed and reopened
                self.bar.display(close=True)
            except TypeError:
                pass
            finally:
                self.bar = DummyBar(total=self.total)

        self.stop_monitors()
        self.data_containers = []
        self.has_graphs = False
        self.log_file = None
        self.telegram = False


    def update(self, n=1, pipeline=None, batch=None):
        """ Update Notifier with new info:
        - fetch up-to-date data from batch, pipeline and monitors
        - set bar postfix
        - draw plots anew
        - update log log_file
        - send notifications to Telegram
        - increment underlying progress bar tracker
        """
        if self.bar.n == 0 or (self.bar.n + 1) % self.frequency == 0 or (self.bar.n == self.bar.total - 1):
            self.timestamps.append(gmtime())

            if self.data_containers:
                self.update_data(pipeline=pipeline, batch=batch)
            self.update_postfix()

            if self.has_graphs:
                self.update_plot(index=self.n_monitors, add_suptitle=True)

            if self.log_file:
                self.update_log_file()

            if self.telegram:
                self.update_telegram()

        self.bar.update(n)

        if self.bar.n == self.bar.total:
            self.close()

    def update_data(self, pipeline=None, batch=None):
        """ Get data from monitor or pipeline. """
        for container in self.data_containers:
            source = container['source']
            if isinstance(source, ResourceMonitor):
                container['data'] = source.data

            elif isinstance(source, str):
                value = pipeline.v(source)
                container['data'] = value

            elif isinstance(source, NamedExpression):
                value = eval_expr(source, pipeline=pipeline, batch=batch)
                container['data'] = value

            elif isinstance(source, (tuple, list, dict)):
                value = eval_expr(source, pipeline=pipeline, batch=batch)
                container['data'] = value

            elif callable(source):
                container['data'] = source()

            else:
                raise TypeError(f'Unknown type of `source`, {type(source)}!')


    def update_postfix(self):
        """ Set the new bar description, if needed. """
        postfix = self.create_description(iteration=-1)

        previous_postfix = self.bar.postfix or ''
        if postfix and not previous_postfix.startswith(postfix):
            self.bar.set_postfix_str(postfix)

    def make_plotter(self, num_graphs=None, layout='horizontal', figsize=None, ncols=None, nrows=None, **kwargs):
        """ Make canvas for plotting graphs. """
        if num_graphs is None:
            num_graphs = len(self.data_containers)

        if ncols is None and nrows is None:
            if layout in ['h', 'horizontal']:
                ncols, nrows = num_graphs, 1
            elif layout in ['v', 'vertical']:
                ncols, nrows = 1, num_graphs
            else:
                raise ValueError(f"Valid `layout` is one of 'h', 'horizontal', 'v', 'vertical', got {layout} instead.")
        else:
            ncols, nrows = plot.infer_ncols_nrows(num_graphs, ncols, nrows, 5)

        if figsize is None:
            figsize = (6 * ncols, 6 * nrows)

        plot_config = {
            'ncols': ncols,
            'nrows': nrows,
            'figsize': figsize,
            'xlabel_size': 15,
            'ylabel_size': 15,
            'tick_labelsize': 15,
            'legend_size': 15,
            'window': 50,
            **kwargs
        }

        return plot(show=False, fix_config=True, **plot_config)

    def update_plot(self, index=0, add_suptitle=False, savepath=None, clear_display=True, show=True,
                    telegram=None, **kwargs):
        """ Draw plots anew. """
        plot_config = PlotConfig(kwargs)

        if clear_display:
            display.clear_output(wait=True)

        if add_suptitle:
            fmt = {
                **self.bar.format_dict,
                'n': self.bar.n + 1,
                'ncols': self.plot_config.get('bar_width', 80),
                'colour': None,
            }
            self.plotter.config['suptitle'] = self.bar.format_meter(**fmt)
            self.plotter.annotate()

        for i, container in enumerate(self.data_containers):
            if i >= index:
                subplot_index = i - index
                subplot_config = plot_config.maybe_index(subplot_index)
                self.update_subplot(container=container, index=subplot_index, **subplot_config)

        if show:
            self.plotter.redraw()

        savepath = savepath or (f'{self.savepath}_{self.bar.n + 1}' if self.savepath is not None else None)

        if savepath:
            self.plotter.save(savepath=savepath)

        telegram = telegram if telegram is not None else self.telegram
        if telegram:
            self.plotter.figure.canvas.flush_events()
            self.telegram_media.send(self.plotter.figure, force_update=True)

    def update_subplot(self, container, index, **kwargs):
        """ Update subplot under given index by data from given container. """
        subplot = self.plotter[index]
        subplot.clear()

        data = container['data']
        source = container['source']
        name = container['name']
        plot_function = container.get('plot_function')
        plot_config = container.get('plot_config', {})
        plot_config = {**plot_config, **kwargs}
        x = np.arange(len(data))[self.slice]
        y = np.array(data)[self.slice]

        if plot_function is not None:
            plot_function(ax=subplot.ax, index=index, x=x, y=y, container=container, notifier=self, **plot_config)
        elif isinstance(source, ResourceMonitor):
            source.plot(plotter=self.plotter, positions=index, **plot_config)
        else:
            source_defaults = {'title': name}
            if isinstance(data, (tuple, list)) or (isinstance(data, np.ndarray) and data.ndim == 1):
                source_defaults['xlabel'] = 'Iteration'

                if 'loss' in name.lower():
                    data = y
                    mode = 'loss'
                    source_defaults['label'] = 'loss'
                else:
                    data = (x, y)
                    mode = 'curve'
                    source_defaults['label'] = None
            elif isinstance(data, np.ndarray) and data.ndim in (2, 3):
                mode = 'image'
            else:
                msg = "Expected data to be 1-dimensional tuple/list/array or 2- or 3-dimensional array."
                if isinstance(data, np.ndarray):
                    msg += f" Got {type(data)} instead of shape {data.shape}"
                else:
                    msg += f" Got {type(data)} instead."
                raise ValueError(msg)

            plot_config = {**source_defaults, **plot_config}
            self.plotter.plot(data=data, mode=mode, positions=index, **plot_config)

    def update_log_file(self):
        """ Update log file on the fly. """
        with open(self.log_file, 'a+', encoding='utf-8') as f:
            print(self.create_message(self.bar.n, self.bar.postfix or ''), file=f)

    def update_telegram(self):
        """ Send a textual notification to a Telegram. """
        fmt = {
            **self.bar.format_dict,
            'n': self.bar.n + 1,
            'ncols': 80,
            'colour': None,
        }
        text = self.bar.format_meter(**fmt).strip()
        idx = text.find('[')

        self.telegram_text.send(f'`{text[:idx]}`\n`{text[idx:]}`')

    # Manual usage of notifier instance
    def plot(self, show=True, savepath=None, **kwargs):
        """ Convenient alias for working with an instance. """
        self.plotter = self.make_plotter(**kwargs)
        self.update_plot(clear_display=False, show=show, savepath=savepath, **kwargs)

    visualize = plot


    def to_file(self, file):
        """ Log all the iteration-wise info (timestamps, descriptions) into file."""
        with open(file, 'w', encoding='utf-8') as f:
            for i in range(self.bar.n):
                description = self.create_description(iteration=i).replace('\n', '  ')
                print(self.create_message(i, description), file=f)

    def __call__(self, iterable):
        if isinstance(iterable, int):
            iterable = range(iterable)

        if self.bar is not None:
            if self.bar.total is None and hasattr(iterable, '__len__'):
                self.total = len(iterable)
            self.make_bar()

        try:
            for item in iterable:
                yield item
                self.update()
            self.close(success=True)
        except: #pylint: disable=bare-except
            self.close(success=False)
            raise

    def __enter__(self):
        return self

    def __exit__(self, _, __, ___):
        self.close()


    def close(self, success=True):
        """ Close the underlying progress bar. """
        #pylint: disable=attribute-defined-outside-init
        if not success:
            self.bar.colour = self.COLOUR_FAILURE

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

            # Extract value from container for the `iteration`
            if isinstance(source, (str, NamedExpression)):
                value = container['data'][iteration]
            elif isinstance(source, list):
                value = container['data'][iteration]
            else:
                continue

            # Trim the value: currently, we work with numbers only
            if 'format' in container:
                desc = container['format'].format(value)
            elif isinstance(value, (float, np.floating)):
                desc = f'{name}={value:2.4f}'
            elif isinstance(value, (int, np.signedinteger,)):
                desc = f'{name}={value:,}'
            else:
                continue
            if desc:
                description.append(desc)
        return ';   '.join(description)

    def create_message(self, iteration, description):
        """ Combine timestamp, iteration and description into one string message. """
        iteration = iteration // self.frequency
        timestamp = strftime("%Y-%m-%d  %H:%M:%S", self.timestamps[iteration])
        return f'{timestamp}     Iteration {iteration:5};    {description}'

    def __getattr__(self, key):
        """ Redirect everything to the underlying bar. """
        if not key.endswith('state__') and key not in self.__dict__ and hasattr(self.bar, key):
            return getattr(self.bar, key)
        raise AttributeError(key)

    def __del__(self):
        """ Extra safety measure to close the underlying bar: helps to prevent tqdm bug. """
        self.close()

    @staticmethod
    def clear():
        """ Clear all the instances. Can help fix tqdm behaviour. """
        # pylint: disable=protected-access
        tqdm._instances.clear()


def notifier(iterable, *args, bar='a', **kwargs):
    """ A convenient wrapper for iterables. """
    return Notifier(bar=bar, *args, **kwargs)(iterable)
