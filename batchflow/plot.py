""" Plot functions. """
from colorsys import rgb_to_hls, hls_to_rgb
from copy import copy
from datetime import datetime
from functools import reduce
from numbers import Number

import numpy as np

from IPython.display import display
from scipy.ndimage import convolve
from matplotlib import pyplot as plt
from matplotlib.colors import ColorConverter, ListedColormap, is_color_like
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, AutoMinorLocator
from mpl_toolkits import axes_grid1

from .utils import to_list



class CycledList(list):
    """ List that repeats itself from desired position (default is 0).

        Examples
        --------
        >>> l = CycledList(['a', 'b', 'c'])
        >>> [l[i] for i in range(9)]
        ['a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c']

        >>> l = CycledList(['a', 'b', 'c', 'd'], cycle_from=2)
        >>> [l[i] for i in range(9)]
        ['a', 'b', 'c', 'd', 'c', 'd', 'c', 'd', 'c']

        >>> l = CycledList(['a', 'b', 'c', 'd', 'e'], cycle_from=-1)
        >>> [l[i] for i in range(9)]
        ['a', 'b', 'c', 'd', 'e', 'e', 'e', 'e', 'e']

        Notes
        -----
        Contrary to `chain(lst, cycle(lst[cycle_from:]))` itertools solution this one is indexable.
    """
    def __init__(self, *args, cycle_from=0, **kwargs):
        self.cycle_from = cycle_from
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx):
        if idx >= len(self):
            pos = self.cycle_from + len(self) * (self.cycle_from < 0)
            if pos < 0:
                raise IndexError(f"List of length {len(self)} is looped from {self.cycle_from} index")
            idx = pos + (idx - pos) % (len(self) - pos)
        return super().__getitem__(idx)

# pylint: disable=invalid-name
class preprocess_and_imshow:
    """ TODO """
    def __init__(self, ax, array, *args, mask_values=(), order_axes=None, vmin=None, vmax=None, **kwargs):
        self.mask_values = to_list(mask_values) if mask_values is not None else []
        self.order_axes = order_axes
        self.vmin, self.vmax = vmin, vmax

        new_array = self._preprocess(array)
        self.im = ax.imshow(new_array, *args, vmin=vmin, vmax=vmax, **kwargs)

    def _preprocess(self, array):
        masks = [array == m if isinstance(m, Number) else m(array) for m in self.mask_values]
        mask = reduce(np.logical_or, masks, np.isnan(array))
        new_array = np.ma.array(array, mask=mask)

        order_axes = self.order_axes[:array.ndim]
        new_array = np.transpose(new_array, order_axes)
        return new_array

    def set_data(self, array):
        """ TODO """
        vmin_new = np.nanmin(array) if self.vmin is None else self.vmin
        vmax_new = np.nanmax(array) if self.vmax is None else self.vmax
        clim = [vmin_new, vmax_new]
        self.im.set_clim(clim)

        new_array = self._preprocess(array)
        self.im.set_data(new_array)

    def __getattr__(self, key):
        if self.im is None:
            return getattr(self, key)
        return getattr(self.im, key)

    def __repr__(self):
        if self.im is None:
            return super().__repr__()
        return self.im.__repr__()


class plot:
    """ Multiple images plotter.

    General parameters
    ----------
    data : np.ndarray or a list of np.ndarray objects or a list of lists of np.ndarray
        If list if flat, 'overlay/separate' logic is handled via `combine` parameter.
        If list is nested, outer level defines subplots order while inner one defines layers order.
        Shape of data items must match chosen plotting mode (see below).
    mode : 'imshow', 'wiggle', 'hist', 'loss'
        If 'imshow' plot given arrays as images.
        If 'wiggle' plot 1d subarrays of given array as signals.
        Subarrays are extracted from given data with fixed step along vertical axis.
        If 'hist' plot histogram of flattened array.
        If 'loss' plot given arrays as loss curves.
    combine : 'overlay', 'separate' or 'mixed'
        Whether overlay images on a single axis, show them on separate ones or use mixed approach.
        Note, that 'wiggle' plot mode is incompatible with `combine='separate'`.
    kwargs :
        - For one of `imshow`, 'wiggle`, `hist` or `loss` methods (depending on chosen mode).
            Parameters and data nestedness levels must match.
            Every param with 'imshow_', 'wiggle_', 'hist_' or 'loss_' prefix is redirected to corresponding method.
            See detailed parameters listings below.
        - For `annotate_axis`.
            Every param with 'title_', 'suptitle_', 'xlabel_', 'ylabel_', 'xticks_', 'yticks_', 'xlim_', 'ylim_',
            colorbar_', 'legend_' or 'grid_' prefix is redirected to corresponding matplotlib method.
            Also 'facecolor', 'set_axisbelow', 'disable_axes' arguments are accepted.

    Parameters for figure creation
    ------------------------------
    figsize : tuple
        Size of displayed figure. If not provided, infered from data shapes.
    facecolor : valid matplotlib color
        Figure background color.

    Parameters for 'imshow' mode
    ----------------------------
    cmap : valid matplotlib colormap or color
        Defines colormap to display single-channel images with.
    alpha : number in (0, 1) range
        Image opacity (0 means fully transparent, i.e. invisible, and 1 - totally opaque).
        Useful when `combine='overlay'`.
    order_axes: tuple
        Order of axes for displayed images.
    mask_values : number or tuple of numbers
        Values that should be masked on image display.
    mask_color : valid matplotlib color
        Color to display masked values with.
    imshow_{parameter} : misc
        Any parameter valid for `plt.imshow`.

    Parameters for 'hist' mode
    ----------------------------
    color : valid matplotlib color
        Defines color to display histogram with.
    alpha : number in (0, 1) range
        Hisotgram opacity (0 means fully transparent, i.e. invisible, and 1 - totally opaque).
        Useful when `combine='overlay'`.
    bins : int
        Number of bins for histogram.
    mask_values : number or tuple of numbers
        Values that should be masked on image display.
    mask_color : valid matplotlib color
        Color to display masked values with.
    hist_{parameter} : misc
        Any parameter valid for `plt.hist`.

    Parameters for 'loss' mode
    ----------------------------
    color : valid matplotlib color
        Defines color to display loss curve with.
    alpha : number in (0, 1) range
        Loss curve opacity (0 means fully transparent, i.e. invisible, and 1 - totally opaque).
        Useful when `combine='overlay'`.
    rolling_mean : number

    rolling_final : number

    loss_{parameter} : misc
        Any parameter valid for `plt.plot`.

    Parameters for axes annotation
    ------------------------------
    {text_object}_label: str
        Value of axes text object.
        Valid objects are 'suptitle', 'title', 'xlabel', 'ylabel', 'legend'.
    {text_object}_color : str or tuple
        Color of axes text object.
        Valid objects are 'suptitle', 'title', 'xlabel', 'ylabel', 'legend'.
        If str, must be valid matplotlib colormap.
        If tuple, must be a valid rgb color.
    {text_object}_size : number
        Size of axes text object.
        Valid objects are 'suptitle', 'title', 'xlabel', 'ylabel', 'legend'.
    colorbar : bool
        Toggle for colorbar.
    colorbar_width : number
        The width of colorbar as a percentage of the subplot width.
    colorbar_pad : number
        The pad of colorbar as a percentage of the subplot width.
    legend_loc : number
        Codes legend position in matplotlib terms (must be from 0-9 range).
    grid: bool
        Grid toggle.
    {object}_{parameter} : misc
        Any parameter with prefix of desired object that is valid for corresponding method:
        title : `plt.set_title`
        suptitle : `plt.suptitle`
        xlabel : `plt.set_xlabel`
        ylabel : `plt.set_ylabel`
        xticks : `plt.set_xticks`
        yticks : `plt.set_yticks`
        tick : `plt.tick_params`
        xlim : `plt.set_xlim`
        ylim : `plt.set_ylim`
        legend : `plt.legend`
        grid : `plt.grid`

    Overall idea
    ------------
    Simply provide data, plot mode and parameters to the `plot` initialization
    and the class takes care of redirecting params to methods they are meant for.

    The logic behind the process is the following:
    1. Parse data:
        - Calculate subplots sizes - look through all data items
          and estimate every subplot shape taking max of all its layers shapes.
        - Put provided arrays into double nested list.
          Nestedness levels define subplot and layer data order correspondingly.
        - Infer images combination mode.
        - Calculate indices corrections for empty subplots.
    2. Parse figure axes if provided, else create them with either parsed parameters or inferred ones.
    3. Obtain default params for chosen mode and merge them with provided params.
    4. For every axis-data pair:
        - If no data provided for axis, set if off.
        - Else filter params relevant for ax, plot data relevant to the ax and annotate it.
    6. Save figure if needed.

    Data display scenarios
    ----------------------
    1. The simplest one if when one provide a single data array.
    2. A more advanced use case is when one provide a list of arrays:
       a. Images are put on same axis and overlaid: data=[image_1, image_2] (combine='overlay' is default);
       b. Images are put on separate axes: data=[image_1, image_2], combine='separate';
    3. The most complex scenario is when one wish to display images in a 'mixed' manner.
       For example, to overlay first two images but to display the third one separately, one must use:
       data = [[image_1, image_2], image_3] (combine='mixed' is set automatically if data is double-nested);

    The order of arrrays inside the double-nested structure basically declares, which of them belong to the same axis
    and therefore should be rendered one over another, and which must be displayed separately.

    Note that in general parameters should resemble data nestedness level.
    That allows binding axes and parameters that correspond each other.
    However, it's possible for parameter to be a single item — in that case it's shared across all subplots and layers.

    Advanced parameters managing
    ----------------------------
    The list of parameters expected by specific plot method is rather short.
    But there is a way to provide parameter to a plot method, even if it's not hard-coded.
    One must use specific prefix for that.
    Address docs of `imshow`, `wiggle`, `hist`, `loss` and `annotate_axis` for details.

    This also allows one to pass arguments of the same name for different plotting steps.
    E.g. `plt.set_title` and `plt.set_xlabel` both require `size` argument.
    Providing `{'size': 30}` in kwargs will affect both title and x-axis labels.
    To change parameter for title only, one can provide {'title_fontsize': 30}` instead.
    """
    def __init__(self, data=None, combine='overlay', mode='imshow', **kwargs):
        """ Plot manager.

        Parses axes from kwargs if provided, else creates them.
        Filters parameters and calls chosen plot method for every axis-data pair.
        """
        data, combine, n_subplots, empty_subplots = self.parse_data(data=data, combine=combine, mode=mode)
        self.data = data
        self.combine = combine
        self.n_subplots = n_subplots
        self.empty_subplots = empty_subplots
        self.idx_fix = np.cumsum([0] + empty_subplots)
        self.fig, self.axes, self.fig_config = self.make_figure(mode=mode, **kwargs)
        self.axes_configs = np.full(len(self.axes), None)
        self.axes_objects = np.full(len(self.axes), None)

        self.plot(mode=mode, **kwargs)

    @staticmethod
    def parse_data(data, combine, mode):
        """ TODO """
        contains_numbers = lambda x: isinstance(x[0], Number)

        def process_tuple(data):
            if mode not in ('curve', 'loss'):
                msg = "Tuple is a valid data item only in modes ('curve', 'loss')."
                raise ValueError(msg)
            return tuple(np.array(item) for item in data)

        def process_array(data):
            if data.ndim > 1 and mode in ('curve', 'loss'):
                msg = f"In `mode={mode}` array must be 1-dimensional, got array with ndim={data.ndim} instead."
                raise ValueError(msg)
            return data

        empty_subplots = []
        n_subplots = 0

        if data is None:
            data_list = [None]
            n_subplots = 1
        elif isinstance(data, tuple):
            data_list = [[process_tuple(data)]]
            n_subplots = 1
        elif isinstance(data, np.ndarray):
            data_list = [[process_array(data)]]
            n_subplots = 1
        elif isinstance(data, list) and contains_numbers(data):
            data_list = [[np.array(data)]]
            n_subplots = 1
        elif isinstance(data, list):
            if any(isinstance(item, list) and not contains_numbers(item) for item in data):
                combine = 'mixed'

            data_list = []
            for item in data:
                empty = 0

                if item is None:
                    if combine == 'overlay':
                        msg = "`None` is a placeholder future subplots. It makes not sense when `combine='overlay'`."
                        raise ValueError(msg)
                    data_item = None
                    empty = 1
                elif isinstance(item, tuple):
                    data_item = [process_tuple(item)]
                elif isinstance(item, np.ndarray):
                    data_item = [process_array(item)]
                elif isinstance(item, list) and contains_numbers(item):
                    data_item = [np.array(item)]
                elif isinstance(item, list):
                    if combine == 'separate':
                        raise ValueError("Data list items cant be lists themselves when `combine='separate'`")
                    data_item = [np.array(subitem) if contains_numbers(subitem) else subitem for subitem in item]
                else:
                    msg = f"Valid data items are None, tuple, array or list of those, got {type(item)} instead."
                    raise ValueError(msg)

                if combine in ('overlay',):
                    data_list.extend(data_item)
                elif combine in ('separate', 'mixed'):
                    data_list.append(data_item)
                    n_subplots += 1
                else:
                    msg = f"Valid combine modes are 'overlay', 'separate', 'mixed', got {combine} instead."
                    raise ValueError(msg)

                empty_subplots.append(empty)

            if combine == 'overlay':
                data_list = [data_list]
                n_subplots = 1

        return data_list, combine, n_subplots, empty_subplots

    def make_default_config(self, mode, ncols=None, nrows=None, **_):
        """ Infer default figure params from shapes of provided data. """
        config = {'tight_layout': True, 'facecolor': 'snow'}

        if mode in ('imshow', 'hist', 'wiggle'):
            default_ncols = 4
        elif mode in ('curve', 'loss'):
            default_ncols = 1

        # Make ncols/nrows
        ceil_div = lambda a, b: -(-a // b)
        if ncols is None and nrows is None:
            ncols = min(default_ncols, self.n_subplots)
            nrows = ceil_div(self.n_subplots, ncols)
        elif ncols is None:
            ncols = ceil_div(self.n_subplots, nrows)
        elif nrows is None:
            nrows = ceil_div(self.n_subplots, ncols)

        config['ncols'], config['nrows'] = ncols, nrows

        return config

    def make_figure(self, mode, axes=None, axis=None, ax=None, **kwargs):
        """ Create figure and axes if needed. """
        axes = axes or axis or ax

        if axes is None:
            config = self.make_default_config(mode=mode, **kwargs)
            subplots_keys = ['figsize', 'facecolor', 'dpi', 'ncols', 'nrows', 'tight_layout', 'gridspec_kw']
            config = self.filter_config(kwargs, subplots_keys, prefix='figure_', save_to=config)

            with plt.ioff():
                fig, axes = plt.subplots(**config)
            axes = to_list(axes)
        else:
            axes = to_list(axes)
            fig = axes[0].figure
            config = {}

            if len(axes) < len(self.n_subplots):
                raise ValueError(f"Not enough axes provided — got ({len(axes)}) for {len(self.n_subplots)} subplots.")

        return fig, axes, config

    def adjust_figsize(self, mode, ncols, nrows, figsize=None, ratio=None, scale=1, max_fig_width=25, **_):
        """ TODO """
        if mode in ('imshow', 'hist', 'wiggle'):
            fig_width = 8 * ncols * scale
        elif mode in ('curve', 'loss'):
            fig_width = 16 * ncols * scale

        # Make figsize
        if figsize is None and ratio is None:
            if mode == 'imshow':
                # redraw figure so that latest plots are applied to obtain correct axes sizes
                self.fig.canvas.draw_idle()

                axes_bboxes = [ax.get_window_extent() for ax in self.axes]
                widths = [bbox.width if bbox is not None else 0 for bbox in axes_bboxes]
                heights = [bbox.height if bbox is not None else 0 for bbox in axes_bboxes]
                mean_height, mean_width = np.mean(heights), np.mean(widths)

                colorbar_bboxes = [ax_objects[0].colorbar.ax.get_window_extent()
                                   if (ax_objects is not None and ax_objects[0].colorbar is not None) else None
                                   for ax_objects in self.axes_objects]
                colorbar_widths = [bbox.width if bbox is not None else 0 for bbox in colorbar_bboxes]
                colorbar_heights = [bbox.height if bbox is not None else 0 for bbox in colorbar_bboxes]
                colorbar_mean_height, colorbar_mean_width = np.mean(colorbar_heights), np.mean(colorbar_widths)

                mean_height += colorbar_mean_height
                mean_width += colorbar_mean_width

                if mean_height == 0 or mean_width == 0:
                    ratio = 1
                else:
                    ratio = (mean_height * nrows) / (mean_width * ncols)

            elif mode == 'hist':
                ratio = 2 / 3 / ncols * nrows

            elif mode == 'wiggle':
                ratio = 1 / ncols * nrows

            elif mode in ('curve', 'loss'):
                ratio = 1 / 3 / ncols * nrows

        if figsize is None:
            fig_height = fig_width * ratio

            if fig_width > max_fig_width:
                fig_width = max_fig_width
                fig_height = fig_width * ratio

            figsize = (fig_width, fig_height)

        self.fig.set_size_inches(figsize)

    def plot(self, data=None, mode='imshow', combine=None, save=False, axes_idx=None, **kwargs):
        """ TODO

        Notes to self: Explain `abs_idx` and `rel_idx`.
        """
        if data is not None:
            data, combine, n_subplots, empty_subplots = self.parse_data(data=data, combine=combine, mode=mode)
            self.data = data
            self.combine = combine
            self.n_subplots = n_subplots
            self.idx_fix = np.cumsum([0] + empty_subplots)

        mode_defaults = getattr(self, f"{mode.upper()}_DEFAULTS")
        self.config = {**self.ANNOTATION_DEFAULTS, **mode_defaults, **kwargs}

        axes_indices = range(len(self.axes)) if axes_idx is None else to_list(axes_idx)
        for rel_idx, abs_idx in enumerate(axes_indices):
            ax = self.axes[abs_idx]
            if rel_idx >= len(self.data) or self.data[rel_idx] is None:
                if self.axes_objects[rel_idx] is None:
                    ax.set_axis_off()
            else:
                plot_method = getattr(self, f"ax_{mode}")
                ax_objects, ax_config = plot_method(ax=ax, idx=rel_idx)
                self.ax_annotate(ax=ax, ax_config=ax_config, ax_objects=ax_objects, idx=rel_idx, mode=mode)

                self.axes_objects[abs_idx] = ax_objects
                self.axes_configs[abs_idx] = ax_config

        figsize_keys = ['ncols', 'nrows', 'figsize', 'ratio', 'scale',
                        'min_fig_width', 'min_fig_height', 'max_fig_width', 'max_fig_height']
        figsize_config = self.filter_config({**self.fig_config, **self.config}, figsize_keys)
        self.adjust_figsize(mode=mode, **figsize_config)

        if save or 'savepath' in kwargs:
            self.save(kwargs)

        return self

    def __call__(self, mode, **kwargs):
        self.plot(mode=mode, **kwargs)

    def _ipython_display_(self):
        self.show()

    def __repr__(self):
        return repr(self.fig).replace('Figure', 'Batchflow Figure')

    ANNOTATION_DEFAULTS = {
        'facecolor': 'snow',
        # text
        'text_color': 'k',
        # suptitle
        'suptitle_size': 25,
        # title
        'title_size': 20,
        # axis labels
        'xlabel': '', 'ylabel': '',
        'xlabel_size': '12', 'ylabel_size': '12',
        # colorbar
        'colorbar': False,
        # grid
        'minor_grid_color': '#CCCCCC',
        'minor_grid_linestyle': '--',
        'major_grid_color': '#CCCCCC',
    }

    def ax_annotate(self, ax, ax_config, ax_objects, idx, mode):
        """ Apply requested annotation functions to given axis with chosen parameters. """
        # pylint: disable=too-many-branches
        text_keys = ['size', 'family', 'color']
        text_params = self.filter_config(ax_config, text_keys, prefix='text_')

        # title
        keys = ['title', 'y'] + text_keys
        params = self.filter_config(ax_config, keys, prefix='title_')
        params['label'] = params.pop('title', params.pop('label', None))
        params = {**text_params, **params}
        if params:
            ax.set_title(**params)

        # suptitle
        keys = ['suptitle', 't', 'y'] + text_keys
        params = self.filter_config(ax_config, keys, prefix='suptitle_')
        params['t'] = params.pop('t', params.pop('suptitle', params.pop('label', None)))
        params = {**text_params, **params}
        if params:
            ax.figure.suptitle(**params)

        # xlabel
        keys = ['xlabel'] + text_keys
        params = self.filter_config(ax_config, keys, prefix='xlabel_', index=idx)
        params = {**text_params, **params}
        if params:
            ax.set_xlabel(**params)

        # ylabel
        keys = ['ylabel'] + text_keys
        params = self.filter_config(ax_config, keys, prefix='ylabel_', index=idx)
        params = {**text_params, **params}
        if params:
            ax.set_ylabel(**params)

        # xticks
        params = self.filter_config(ax_config, ['xticks'], prefix='xticks_', index=idx)
        if 'xticks' in params:
            params['ticks'] = params.get('ticks', params.pop('xticks'))
        if params:
            ax.set_xticks(**params)

        # yticks
        params = self.filter_config(ax_config, ['yticks'], prefix='yticks_', index=idx)
        if 'yticks' in params:
            params['ticks'] = params.get('ticks', params.pop('yticks'))
        if params:
            ax.set_yticks(**params)

        # ticks
        keys = ['labeltop', 'labelright', 'labelcolor', 'direction']
        params = self.filter_config(ax_config, keys, prefix='tick_', index=idx)
        if params:
            ax.tick_params(**params)

        # xlim
        params = self.filter_config(ax_config, ['xlim'], prefix='xlim_', index=idx)
        if 'xlim' in params:
            params['left'] = params.get('left', params.pop('xlim'))
        if params:
            ax.set_xlim(**params)

        # ylim
        params = self.filter_config(ax_config, ['ylim'], prefix='ylim_', index=idx)
        if 'ylim' in params:
            params['bottom'] = params.get('bottom', params.pop('ylim'))
        if params:
            ax.set_ylim(**params)

        # colorbar
        if any(to_list(self.config['colorbar'])):
            keys = ['colorbar', 'width', 'pad', 'fake', 'ax_objects']
            params = self.filter_config(ax_config, keys, prefix='colorbar_', index=idx)
            params['ax_image'] = ax_objects[0]
            # if colorbar is disabled for subplot, add param to plot fake axis instead to keep proportions
            params['fake'] = not params.pop('colorbar', True)
            self.add_colorbar(**params)

        # legend
        legend = self.filter_config(ax_config, 'legend')
        keys = ['label', 'size', 'loc', 'ha', 'va']
        params = self.filter_config(ax_config, keys, prefix='legend_')
        if legend or params:
            if mode == 'imshow':
                color = self.filter_config(ax_config, keys=['color', 'cmap'], prefix='legend_')
                params['color'] = list(color.values())[0]
            elif mode in ('curve', 'loss'):
                params['handles'] = ax_objects
            self.add_legend(ax, mode=mode, **params)

        # grid
        grid = self.filter_config(ax_config, 'grid', index=idx)
        grid_keys = ['color', 'linestyle', 'freq']

        minor_params = self.filter_config(ax_config, grid_keys, prefix='minor_grid_', index=idx)
        if grid in ('minor', 'both') and minor_params:
            self.add_grid(ax, grid_type='minor', **minor_params)

        major_params = self.filter_config(ax_config, grid_keys, prefix='major_grid_', index=idx)
        if grid in ('major', 'both') and minor_params:
            self.add_grid(ax, grid_type='major', **major_params)

        facecolor = ax_config.get('facecolor', None)
        if facecolor is not None:
            ax.set_facecolor(facecolor)

        ax.set_axisbelow(ax_config.get('set_axisbelow', False))

        if ax_config.get('disable_axes'):
            ax.set_axis_off()
        elif not ax.axison:
            ax.set_axis_on()

    def show(self):
        display(self.fig)

    def save(self, kwargs):
        """ Save plot. """
        default_params = {
            'savepath': datetime.now().strftime('%Y-%m-%d_%H:%M:%S.png'),
            'bbox_inches': 'tight',
            'pad_inches': 0,
            'dpi': 100
        }

        save_keys = ['savepath', 'bbox_inches', 'pad_inches', 'dpi']
        save_params = self.filter_config(kwargs, save_keys, prefix='save_')
        save_params = {**default_params, **save_params}
        savepath = save_params.pop('savepath')

        self.fig.savefig(fname=savepath, **save_params)


    # Rendering methods
    MASK_COLORS = ['firebrick', 'mediumseagreen', 'thistle', 'darkorange', 'navy', 'gold',
                   'red', 'turquoise', 'darkorchid', 'darkkhaki', 'royalblue', 'yellow',
                   'chocolate', 'forestgreen', 'lightpink', 'darkslategray', 'deepskyblue', 'wheat']

    IMSHOW_DEFAULTS = {
        # image
        'cmap': CycledList(['Greys_r', *MASK_COLORS], cycle_from=1),
        'colorbar_width': 5,
        'colorbar_pad': None,
        # ticks
        'labeltop': True,
        'labelright': True,
        'direction': 'inout',
        # axes order
        'order_axes': (0, 1, 2),
        # values masking
        'mask_color': (0, 0, 0, 0),
        # grid
        'grid': False,
    }

    def ax_imshow(self, ax, idx):
        """ TODO """
        subplot_idx = None if self.combine == 'overlay' else idx - self.idx_fix[idx]
        config = self.filter_config(self.config, index=subplot_idx)
        images = []

        for image_idx, image in enumerate(self.data[idx]):

            layer_idx = image_idx
            if self.combine == 'separate':
                layer_idx += idx - self.idx_fix[idx]

            imshow_keys = ['vmin', 'vmax', 'interpolation', 'alpha', 'extent', 'order_axes', 'mask_values']
            imshow_params = self.filter_config(config, imshow_keys, prefix='imshow_', index=layer_idx)

            # Assemble colormap from given parameters
            cmap = self.filter_config(config, 'cmap', index=layer_idx)
            # If a single color provided, prepend 'white' color, so that a resulting list defines binary colormap
            if is_color_like(cmap):
                cmap = ['white', cmap]
            # If a list of colors provided in `cmap` argument convert it into a colormap
            if isinstance(cmap, list):
                cmap = self.make_cmap(colors=cmap)
            else:
                cmap = copy(plt.get_cmap(cmap))
            # Set a color for nan/masked values display to colormap if provided
            mask_color = self.filter_config(config, 'mask_color', index=layer_idx)
            cmap.set_bad(color=mask_color)
            # Add created cmap to a dict of imshow params
            imshow_params['cmap'] = cmap

            # Add `0` to a list of values that shouldn't be displayed if image is a binary mask
            if tuple(np.unique(image)) in [(0, ), (0, 1)]:
                imshow_params['mask_values'] = 0
                imshow_params['vmin'] = 0

            # Use a proxy for imshow calls that fixes data preprocessing parameters
            # and re-applies them to axes image before `set_data` calls
            image = preprocess_and_imshow(ax=ax, array=image, **imshow_params)
            images.append(image)

        return images, config


    HIST_DEFAULTS = {
        # hist
        'bins': 50,
        'color': CycledList(MASK_COLORS),
        'alpha': 0.8,
        # axis labels
        'xlabel': 'Values',
        'ylabel': 'Counts',
        # common
        'set_axisbelow': True,
        'colorbar': False,
        # grid
        'grid': 'major',
    }

    def ax_hist(self, ax, idx):
        """ TODO """
        subplot_idx = None if self.combine == 'overlay' else idx - self.idx_fix[idx]
        config = self.filter_config(self.config, index=subplot_idx)
        objects = []

        for array_idx, array in enumerate(self.data[idx]):
            layer_idx = array_idx
            if self.combine == 'separate':
                layer_idx += idx - self.idx_fix[idx]

            hist_keys = ['bins', 'color', 'alpha', 'label']
            hist_params = self.filter_config(config, hist_keys, prefix='hist_', index=layer_idx)

            mask_values = self.filter_config(config, 'mask_values', index=layer_idx) or []
            masks = [array == m if isinstance(m, Number) else m(array) for m in mask_values]
            mask = reduce(np.logical_or, masks, np.isnan(array))
            new_array = np.ma.array(array, mask=mask).flatten()

            _, _, obj = ax.hist(new_array, **hist_params)
            objects.append(obj)

        return objects, config


    CURVE_COLORS = ['cornflowerblue', 'sandybrown', 'lightpink', 'mediumseagreen', 'thistle', 'firebrick',
                    'forestgreen', 'navy', 'gold', 'red', 'turquoise', 'darkorchid',
                    'darkkhaki', 'royalblue', 'yellow', 'chocolate', 'darkslategray', 'wheat']

    CURVE_DEFAULTS = {
        # curve
        'color': CycledList(CURVE_COLORS),
        # axis labels
        'xlabel': 'x',
        'ylabel': 'y',
        # common
        'colorbar': False,
        # grid
        'grid': 'both',
    }

    def ax_curve(self, ax, idx):
        """ TODO """
        subplot_idx = None if self.combine == 'overlay' else idx - self.idx_fix[idx]
        config = self.filter_config(self.config, index=subplot_idx)
        objects = []

        for array_idx, arrays in enumerate(self.data[idx]):
            if isinstance(arrays, np.ndarray):
                if arrays.ndim == 1:
                    x = range(len(arrays))
                    y = arrays
                else:
                    x, y = arrays
            elif isinstance(arrays, tuple):
                if len(arrays) == 1:
                    x = range(len(arrays))
                    y = arrays
                else:
                    x, y = arrays
            else:
                raise ValueError('Valid data object is either np.array or tuple of np.arrays')

            layer_idx = array_idx
            if self.combine == 'separate':
                layer_idx += idx - self.idx_fix[idx]

            curve_keys = ['color', 'linestyle', 'alpha', 'label']
            curve_params = self.filter_config(config, curve_keys, prefix='curve_', index=layer_idx)
            curve_line = ax.plot(x, y, **curve_params)
            objects.extend(curve_line)

            # Change scale of axis, if needed
            if config.get('log'):
                ax.set_yscale('log')

        return objects, config


    LOSS_DEFAULTS = {
        # main
        'window': 20,
        'final_window': 50,
        # curve
        'color': CycledList(CURVE_COLORS[::2]),
        # learning rate
        'lr_color': CycledList(CURVE_COLORS[1::2]),
        # title
        'title_label': 'Loss values and learning rate',
        # axis labels
        'xlabel': 'Iterations', 'ylabel': 'Loss',
        # common
        'colorbar': False,
        # grid
        'grid': 'both',
        'minor_grid_y_n': 4,
        # legend
        'legend': True,
    }

    def ax_loss(self, ax, idx):
        """ TODO """
        subplot_idx = None if self.combine == 'overlay' else idx - self.idx_fix[idx]
        config = self.filter_config(self.config, index=subplot_idx)
        objects = []

        lr_ax = None
        for array_idx, arrays in enumerate(self.data[idx]):
            if isinstance(arrays, np.ndarray):
                if arrays.ndim == 1:
                    loss = arrays
                    lr = None
                else:
                    loss, lr = arrays
            elif isinstance(arrays, tuple):
                if len(arrays) == 1:
                    loss = arrays[0]
                    lr = None
                else:
                    loss, lr = arrays
            else:
                raise ValueError('Valid data object is either np.array or tuple of np.arrays')

            layer_idx = array_idx
            if self.combine == 'separate':
                layer_idx += idx - self.idx_fix[idx]

            label = self.filter_config(config, 'label') or f'loss №{array_idx}'
            loss_label = label + f' ⟶ {loss[-1]:2.3f}'
            final_window = self.filter_config(self.config, 'final_window')
            if final_window is not None:
                final = np.mean(loss[-final_window:]) #pylint: disable=invalid-unary-operand-type
                loss_label += f"\nmean over last {final_window} iterations={final:2.3f}"

            curve_keys = ['color', 'linestyle', 'linewidth', 'alpha']
            loss_params = self.filter_config(config, curve_keys, prefix='curve_', index=layer_idx)
            loss_line = ax.plot(loss, label=loss_label, **loss_params)
            objects.extend(loss_line)

            window = self.filter_config(config, 'window', index=layer_idx)
            if window:
                averaged = convolve(loss, np.ones(window), mode='nearest') / window
                mean_color = self.scale_lightness(loss_params['color'], scale=.5)
                averaged_loss_label = label + ' running mean'
                average_line = ax.plot(averaged, label=averaged_loss_label, color=mean_color, linestyle='--')
                objects.extend(average_line)

            # Change scale of axis, if needed
            if config.get('log_loss'):
                ax.set_yscale('log')

            if lr is not None:
                if lr_ax is None:
                    lr_ax = ax.twinx()
                lr_label = f'learning rate №{array_idx} ⟶ {lr[-1]:.0e}'
                lr_params = self.filter_config(config, curve_keys, prefix='lr_', index=layer_idx)
                lr_line = lr_ax.plot(lr, label=lr_label, **lr_params)
                lr_ax.set_ylabel('Learning rate', fontsize=12)
                objects.extend(lr_line)

            if lr is not None and config.get('log_lr'):
                lr_ax.set_yscale('log')

        return objects, config

    # Supplementary methods

    @staticmethod
    def filter_config(params, keys=None, prefix='', index=None, save_to=None):
        """ Make a subdictionary of parameters with required keys.

        Parameter are retrieved if:
        a. It is explicitly requested (via `keys` arg).
        b. Its name starts with given prefix (defined by `prefix` arg).

        Parameters
        ----------
        params : dict
            Arguments to filter.
        keys : str or sequence
            Key(s) to retrieve. If str, return key value.
            If list — return dict of pairs (key, value) for every existing key.
        prefix : str, optional
            Arguments with keys starting with given prefix will also be retrieved.
            Defaults to `''`, i.e. no prefix used.
        index : int
            Index of argument value to retrieve.
            If none provided, get whole argument value.
            If value is non-indexable, get it without indexing.
        save_to : dict
            Config to populate with filtered items. If none provided, empty dict is used.
        """
        # get value by index if it is requested and value is a list
        maybe_index = lambda value, index: value[index] if index is not None and isinstance(value, list) else value

        if isinstance(keys, str):
            value = params.get(keys, None)
            return maybe_index(value, index)

        if keys is None:
            keys = list(params.keys())
        elif prefix:
            keys += [key.split(prefix)[1] for key in params if key.startswith(prefix)]

        result = {} if save_to is None else save_to

        for key in keys:
            value = params.get(prefix + key, params.get(key))
            if value is not None:
                result[key] = maybe_index(value, index)

        return result

    @staticmethod
    def make_cmap(colors):
        """ Make colormap from provided color/colors list. """
        colors = [ColorConverter().to_rgb(color) if isinstance(color, str) else color for color in to_list(colors)]
        cmap = ListedColormap(colors)
        return cmap

    @staticmethod
    def scale_lightness(color, scale):
        """ Make new color with modified lightness from existing. """
        if isinstance(color, str):
            color = ColorConverter.to_rgb(color)
        hue, light, saturation = rgb_to_hls(*color)
        new_color = hls_to_rgb(h=hue, l=min(1, light * scale), s=saturation)
        return new_color

    @staticmethod
    def add_colorbar(ax_image, width=5, pad=None, color='black', fake=False):
        """ Append colorbar to the image on the right. """
        divider = axes_grid1.make_axes_locatable(ax_image.axes)
        pad = width * 1.5 if pad is None else pad
        cax = divider.append_axes("right", size=f"{width}%", pad=f"{pad}%")
        if fake:
            cax.set_axis_off()
        else:
            colorbar = ax_image.axes.figure.colorbar(ax_image, cax=cax)
            colorbar.ax.yaxis.set_tick_params(color=color)
            ax_image.axes.created_colorbar = colorbar

    @staticmethod
    def add_legend(ax, mode, handles=None, label=None, color=None, size=10, ha=None, va=None, **kwargs):
        """ TODO Add patches to legend. All invalid colors are filtered.

        Notes to self: Rewrite doc, explain line/patches parametrization.
        """
        legend = ax.get_legend()
        old_handles = getattr(legend, 'legendHandles', [])
        texts = getattr(legend, 'get_texts', lambda: [])()
        old_labels = [t._text for t in texts] # pylint: disable=protected-access

        if mode in ('imshow', 'wiggle'):
            colors = [color for color in to_list(color) if is_color_like(color)]
            new_handles = [Patch(color=color) for color in colors]
            new_labels = to_list(label)
        elif mode in ('curve', 'loss'):
            new_handles = handles
            new_labels = [line.get_label() for line in handles]

        kwargs['handles'] = old_handles + new_handles
        kwargs['labels'] = old_labels + new_labels

        legend = ax.legend(prop={'size': size}, **kwargs)

        if ha is not None:
            _ = [text.set_ha(ha) for text in legend.get_texts()]
        if va is not None:
            _ = [text.set_ha(va) for text in legend.get_texts()]

    @staticmethod
    def add_grid(ax, grid_type, x_n=None, y_n=None, zorder=0, **kwargs):
        """ TODO """
        if grid_type == 'minor':
            locator = AutoMinorLocator
        elif grid_type == 'major':
            locator = MaxNLocator

        if x_n is not None:
            set_locator = getattr(ax.xaxis, f'set_{grid_type}_locator')
            set_locator(locator(x_n))

        if y_n is not None:
            set_locator = getattr(ax.yaxis, f'set_{grid_type}_locator')
            set_locator(locator(y_n))

        ax.grid(which=grid_type, zorder=zorder, **kwargs)



def plot_image(data, **kwargs):
    """ Shorthand for image plotting. """
    return plot(data, mode='imshow', **kwargs)

def plot_hist(data, **kwargs):
    """ Shorthand for histogram plotting. """
    return plot(data, mode='hist', **kwargs)

def plot_curve(data, **kwargs):
    """ Shorthand for curve plotting. """
    return plot(data, mode='curve', **kwargs)

def plot_loss(data, **kwargs):
    """ Shorthand for loss plotting. """
    return plot(data, mode='loss', **kwargs)
