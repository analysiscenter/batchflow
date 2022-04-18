""" Plot functions. """
from colorsys import rgb_to_hls, hls_to_rgb
from copy import copy
from datetime import datetime
from functools import reduce
from itertools import cycle
from numbers import Number

import numpy as np

from scipy.ndimage import convolve
from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.colors import ColorConverter, ListedColormap, is_color_like
from matplotlib.patches import Patch, Rectangle
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerBase
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
    """ Proxy class that handles data transformations preceding `imshow` and `set_data` calls.

    Initally, is must be instantiated with parameters valid for `plt.imshow` plus a few of its own (see below).
    On that stage two things happen:
    - Provided data is preprocessed in accordance with given parameters and those parameters are saved.
    - Method `plt.imshow` is called on transformed data and resulted `AxesImage` instance is also saved.

    After initialization, class instance re-applies same data preprocessing when `set_data` is called on it.

    Parameters
    ----------
    ax : `matplotlib.axes.Axes`
        Axis to call `imshow` upon.
    array : `np.ndarray`
        Array to display initially.
    mask_values : iterable of numbers
        Values in data array to mask (diplayed with colormap's bad color).
    order_axes : tuple of numbers
        Parameter for data transpose.
    vmin, vmax : numbers
        For visualized data normalization.
    args, kwargs : misc
        For `plt.imshow`.

    Notes
    -----
    Used to apply consistent data transformation across `set_data` calls during interactive events.
    """
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
        """ Apply data transformations with saved parameters and call `set_data` on `AxesImage` instance. """
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


class ColorMappingHandler(HandlerBase):
    """ Handler mapping for `plt.legend` that transforms colormap name to patches collection of its colors. """
    def __init__(self, n_segments=8, **kwargs):
        super().__init__(**kwargs)
        self.n_segments = n_segments

    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        """ Create a rectangle consisting of sequential patches with colors of colormap. """
        _ = legend, fontsize
        segment_width = width / self.n_segments
        cmap = plt.get_cmap(orig_handle)

        segments = []
        for segment_index in range(self.n_segments):
            xy = [xdescent + segment_index * segment_width, ydescent]
            facecolor = cmap(segment_index / self.n_segments)
            segment = Rectangle(xy, segment_width, height, facecolor=facecolor, transform=trans)
            segments.append(segment)
        patch = PatchCollection(segments, match_original=True, edgecolor=None, cmap=orig_handle)

        return [patch]


class plot:
    """ Multiple images plotter.

    Overall idea
    ------------
    General idea is to display graphs for provided data while passing other keyword arguments to corresponding
    `matplotlib` functions (e.g. `figsize` goes to figure initialization, `title` goes to `plt.set_title`, etc.).

    The logic behind the process is the following:
    1. Parse data:
        - Calculate subplots sizes - look through all data items
          and estimate every subplot shape taking max of all its layers shapes.
        - Put provided arrays into double nested list.
          Nestedness levels define subplot and layer data order correspondingly.
        - Infer images combination mode.
        - Calculate indices corrections for empty subplots.
    2. Parse figure axes if provided, else create them with either parsed parameters or inferred ones.
    3. Obtain default config for chosen mode and merge them with provided config.
    4. For every axis-data pair:
        - If no data provided for axis, set it off.
        - Else filter config relevant for ax, plot data relevant to the ax and annotate it.
    6. Save figure if needed.

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
        """ Plot manager. """
        self.fig = None
        self.axes = []
        self.fig_config = {}
        self.axes_configs = []
        self.axes_objects = []

        self.plot(data=data, combine=combine, mode=mode, **kwargs)

    @staticmethod
    def process_tuple(data, mode):
        """ Validate that tuple data item is provided with correct plot mode and convert its objects to arrays. """
        if mode not in ('curve', 'loss'):
            msg = "Tuple is a valid data item only in modes ('curve', 'loss')."
            raise ValueError(msg)
        return tuple(np.array(item) for item in data)

    @staticmethod
    def process_array(data, mode):
        """ Validate that data dimensionality is correct for given plot mode. """
        if data.ndim > 1 and mode in ('curve', 'loss'):
            msg = f"In `mode={mode}` array must be 1-dimensional, got array with ndim={data.ndim} instead."
            raise ValueError(msg)
        return data

    @classmethod
    def parse_data(cls, data, combine, mode):
        """ Validate input data and put it into a double-nested list.

        First level of nestedness corresponds to subplots indexing.
        Second level of nestedness corresponds to layers indexing.

        So `[array_0, array_1]` is converted to:
        - `[[array_0, array_1]] when `combine='overlay'`
        - `[[array_0], [array_1]] when `combine='separate'`
        """
        contains_numbers = lambda x: isinstance(x[0], Number)

        empty_subplots = []
        n_subplots = 0

        if data is None:
            data_list = [None]
            n_subplots = 1
        elif isinstance(data, tuple):
            data_list = [[cls.process_tuple(data=data, mode=mode)]]
            n_subplots = 1
        elif isinstance(data, np.ndarray):
            data_list = [[cls.process_array(data=data, mode=mode)]]
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
                    data_item = [cls.process_tuple(data=item, mode=mode)]
                elif isinstance(item, np.ndarray):
                    data_item = [cls.process_array(data=item, mode=mode)]
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

    def make_default_config(self, mode, n_subplots, shapes, ncols=None, ratio=None, scale=1, max_fig_width=25,
                            nrows=None, xlim=(None, None), ylim=(None, None), **kwargs):
        """ Infer default figure params from shapes of provided data. """
        config = {'tight_layout': True, 'facecolor': 'snow'}

        if mode in ('imshow', 'hist', 'wiggle'):
            default_ncols = 4
        elif mode in ('curve', 'loss'):
            default_ncols = 1

        # Make ncols/nrows
        ceil_div = lambda a, b: -(-a // b)
        if ncols is None and nrows is None:
            ncols = min(default_ncols, n_subplots)
            nrows = ceil_div(n_subplots, ncols)
        elif ncols is None:
            ncols = ceil_div(n_subplots, nrows)
        elif nrows is None:
            nrows = ceil_div(n_subplots, ncols)

        config['ncols'], config['nrows'] = ncols, nrows

        if mode in ('imshow', 'hist', 'wiggle'):
            fig_width = 8 * ncols * scale
        elif mode in ('curve', 'loss'):
            fig_width = 16 * ncols * scale

        # Make figsize
        if ratio is None:
            if mode == 'imshow':
                if not isinstance(xlim, list):
                    xlim = [xlim] * n_subplots
                if not isinstance(ylim, list):
                    ylim = [ylim] * n_subplots

                widths = []
                heights = []
                for idx, shape in enumerate(shapes):
                    if shape is None:
                        continue

                    order_axes = self.filter_config(kwargs, 'order_axes', index=idx)
                    order_axes = order_axes or self.IMSHOW_DEFAULTS['order_axes']

                    min_height = ylim[idx][0] or 0
                    max_height = ylim[idx][1] or shape[order_axes[0]]
                    subplot_height = abs(max_height - min_height)
                    heights.append(subplot_height)

                    min_width = xlim[idx][0] or shape[order_axes[1]]
                    max_width = xlim[idx][1] or 0
                    subplot_width = abs(max_width - min_width)
                    widths.append(subplot_width)

                mean_height, mean_width = np.mean(heights), np.mean(widths)
                if mean_height == 0 or mean_width == 0:
                    ratio = 1
                else:
                    ratio = (mean_height * 1.05 * nrows) / (mean_width * 1.05 * ncols)

            elif mode == 'hist':
                ratio = 2 / 3 / ncols * nrows

            elif mode == 'wiggle':
                ratio = 1 / ncols * nrows

            elif mode in ('curve', 'loss'):
                ratio = 1 / 3 / ncols * nrows

        fig_height = fig_width * ratio

        if fig_width > max_fig_width:
            fig_width = max_fig_width
            fig_height = fig_width * ratio

        config['figsize'] = (fig_width, fig_height)

        return config

    def make_figure(self, mode, n_subplots, shapes, axes=None, axis=None, ax=None, figure=None, fig=None, **kwargs):
        """ Create figure and axes if needed. """
        axes = axes or axis or ax
        fig = figure or fig
        if axes is None and fig is not None:
            axes = fig.axes

        # enable figsize adjustment by default for `imshow` mode
        adjust_figsize = mode == 'imshow'

        if axes is None:
            default_config = self.make_default_config(mode=mode, n_subplots=n_subplots, shapes=shapes, **kwargs)
            subplots_keys = ['figsize', 'facecolor', 'dpi', 'ncols', 'nrows', 'tight_layout', 'gridspec_kw']
            config = self.filter_config(kwargs, subplots_keys, prefix='figure_')
            if 'figsize' in config:
                adjust_figsize = False # disable figsize adjustment if explicit figsize provided
            config = {**default_config, **config}

            with plt.ioff():
                fig, axes = plt.subplots(**config)
            axes = to_list(axes)
        else:
            axes = to_list(axes)
            fig = axes[0].figure
            config = {}

            if len(axes) < n_subplots:
                raise ValueError(f"Not enough axes provided — got ({len(axes)}) for {n_subplots} subplots.")

        return fig, axes, config, adjust_figsize

    def get_bbox(self, obj):
        """ Get object bounding box in inches. """
        renderer = self.fig.canvas.get_renderer()
        transformer = self.fig.dpi_scale_trans.inverted()
        return obj.get_window_extent(renderer=renderer).transformed(transformer)

    def adjust_figsize(self):
        """ Look through axes' annotation objects and add figsize corrections for their widths and heights. """
        ncols = self.fig_config['ncols']
        nrows = self.fig_config['nrows']
        fig_width, fig_height = self.fig_config['figsize']

        extra_width = 0
        extra_height = 0
        if 'suptitle' in self.fig_objects:
            suptitle_obj = self.fig_objects['suptitle']
            suptitle_height = self.get_bbox(suptitle_obj).height
            extra_height += suptitle_height

        ax_widths = []
        ax_heights = []
        for ax, ax_objects in zip(self.axes, self.axes_objects):
            width = 0
            height = 0
            if ax_objects is not None:
                ax_bbox = self.get_bbox(ax)

                if 'title' in ax_objects:
                    title_obj = ax_objects['title']
                    title_height = self.get_bbox(title_obj).height
                    height += title_height

                xticks_objects = ax.get_xticklabels()
                first_xtick_bbox = self.get_bbox(xticks_objects[0]) # first lower xticklabel bbox
                lower_xticks_height = max(0, ax_bbox.y0 - first_xtick_bbox.y0)
                height += lower_xticks_height

                last_xtick_bbox = self.get_bbox(xticks_objects[-1])
                # if last xticklabel bbox is heigher that the first, there are labels atop of the subplot
                if first_xtick_bbox.y0 != last_xtick_bbox.y0:
                    upper_xticks_height = max(0, last_xtick_bbox.y1 - ax_bbox.y1)
                    height += upper_xticks_height

                if 'xlabel' in ax_objects:
                    xlabel_obj = ax_objects['xlabel']
                    xlabel_height = self.get_bbox(xlabel_obj).height
                    height += xlabel_height

                yticks_objects = ax.get_yticklabels()
                first_ytick_bbox = self.get_bbox(yticks_objects[0]) # first lower xticklabel bbox
                lower_yticks_width = max(0, ax_bbox.x0 - first_ytick_bbox.x0)
                width += lower_yticks_width

                last_ytick_bbox = self.get_bbox(yticks_objects[-1])
                # if last yticklabel bbox is righter that the first, there are labels to the right of the subplot
                if first_ytick_bbox.x0 != last_ytick_bbox.x0:
                    right_yticks_width = max(0, last_ytick_bbox.x1 - ax_bbox.x1)
                    width += right_yticks_width

                if 'ylabel' in ax_objects:
                    ylabel_obj = ax_objects['ylabel']
                    ylabel_width = self.get_bbox(ylabel_obj).width
                    width += ylabel_width

            ax_widths.append(width)
            ax_heights.append(height)

        ax_widths = np.array(ax_widths).reshape(nrows, ncols)
        extra_width += ax_widths.max(axis=1).sum()

        ax_heights = np.array(ax_heights).reshape(nrows, ncols)
        extra_height += ax_heights.max(axis=0).sum()

        new_figsize = (fig_width + extra_width, fig_height + extra_height)
        self.fig.set_size_inches(new_figsize)

    def plot(self, data=None, combine='overlay', mode='imshow', save=False, show=False, adjust_figsize=False, **kwargs):
        """ TODO

        Parses axes from kwargs if provided, else creates them.
        Filters parameters and calls chosen plot method for every axis-data pair.

        Notes to self: Explain `abs_idx` and `rel_idx`.
        """
        data, combine, n_subplots, empty_subplots = self.parse_data(data=data, combine=combine, mode=mode)

        if self.fig is None:
            if mode == 'imshow':
                shapes = [subplot_data[0].shape if subplot_data is not None else None for subplot_data in data]
            else:
                shapes = None
            self.fig, self.axes, self.fig_config, adjust_figsize = self.make_figure(mode=mode, n_subplots=n_subplots,
                                                                                    shapes=shapes, **kwargs)
            self.axes_configs = np.full(len(self.axes), None)
            self.axes_objects = np.full(len(self.axes), None)

        self.config = {**self.ANNOTATION_DEFAULTS, **kwargs}

        ax = kwargs.get('axes') or kwargs.get('axis') or kwargs.get('ax')
        if ax is None:
            axes_indices = range(len(self.axes))
        elif isinstance(ax, int):
            axes_indices = [ax]
        elif isinstance(ax, list) and all(isinstance(item, int) for item in ax):
            axes_indices = ax
        else:
            msg = f"When figure already created one can only specify ax indices to use, got {type(ax)} instead."
            raise ValueError(msg)

        mode_defaults = getattr(self, f"{mode.upper()}_DEFAULTS")

        idx_fixes = np.cumsum([0] + empty_subplots)

        for rel_idx, abs_idx in enumerate(axes_indices):
            subplot_ax = self.axes[abs_idx]

            if rel_idx >= len(data) or data[rel_idx] is None:
                if self.axes_objects[rel_idx] is None:
                    subplot_ax.set_axis_off()
            else:
                plot_method = getattr(self, f"ax_{mode}")

                ax_data = data[rel_idx]
                subplot_idx = None if combine == 'overlay' else rel_idx - idx_fixes[rel_idx]
                ax_config = self.filter_config(self.config, index=subplot_idx)
                ax_config = {**mode_defaults, **ax_config}

                idx_fix = rel_idx - idx_fixes[rel_idx] if combine == 'separate' else None

                ax_objects, ax_config = plot_method(data=ax_data, ax=subplot_ax, config=ax_config, idx_fix=idx_fix)
                ax_objects, ax_config = self.ax_annotate(ax=subplot_ax, ax_config=ax_config, ax_objects=ax_objects,
                                                         index=idx_fix, mode=mode)

                self.axes_objects[abs_idx] = ax_objects
                self.axes_configs[abs_idx] = ax_config

        self.fig_objects = self.fig_annotate()

        if adjust_figsize:
            self.adjust_figsize()

        if show:
            self.show()

        if save or 'savepath' in kwargs:
            self.save(**kwargs)

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
        'suptitle_size': 30,
        # title
        'title_size': 25,
        # axis labels
        # 'xlabel': '', 'ylabel': '',
        'xlabel_size': '12', 'ylabel_size': '12',
        # colorbar
        'colorbar': False,
        # grid
        'minor_grid_color': '#CCCCCC',
        'minor_grid_linestyle': '--',
        'major_grid_color': '#CCCCCC',
    }

    def ax_annotate(self, ax, ax_config, ax_objects, index, mode):
        """ Apply requested annotation functions to given axis with chosen parameters. """
        # pylint: disable=too-many-branches
        text_keys = ['size', 'family', 'color']
        text_config = self.filter_config(ax_config, text_keys, prefix='text_')

        # title
        keys = ['title', 'y']
        title_config = self.filter_config(ax_config, keys, prefix='title_')
        label = None
        if 'label' in title_config:
            label = title_config.pop('label')
        if 'title' in title_config:
            label = title_config.pop('title')
        title_config['label'] = label
        title_config = {**text_config, **title_config}
        if title_config:
            ax_objects['title'] = ax.set_title(**title_config)

        # xlabel
        keys = ['xlabel']
        xlabel_config = self.filter_config(ax_config, keys, prefix='xlabel_', index=index)
        xlabel_config = {**text_config, **xlabel_config}
        if xlabel_config and 'xlabel' in xlabel_config:
            ax_objects['xlabel'] = ax.set_xlabel(**xlabel_config)

        # ylabel
        keys = ['ylabel']
        ylabel_config = self.filter_config(ax_config, keys, prefix='ylabel_', index=index)
        ylabel_config = {**text_config, **ylabel_config}
        if ylabel_config and 'ylabel' in ylabel_config:
            ax_objects['ylabel'] = ax.set_ylabel(**ylabel_config)

        # xticks
        xticks_config = self.filter_config(ax_config, [], prefix='xticks_', index=index)
        ticks = self.filter_config(ax_config, 'ticks', index=index)
        xticks = self.filter_config(ax_config, 'xticks', index=index)
        xticks = ticks if ticks is not None else xticks
        if xticks is not None:
            xticks_config['ticks'] = xticks
        if xticks_config:
            ax.set_xticks(**xticks_config)

        # yticks
        yticks_config = self.filter_config(ax_config, [], prefix='yticks_', index=index)
        ticks = self.filter_config(ax_config, 'ticks', index=index)
        yticks = self.filter_config(ax_config, 'yticks', index=index)
        yticks = ticks if ticks is not None else yticks
        if yticks is not None:
            yticks_config['ticks'] = yticks
        if yticks_config:
            ax.set_yticks(**yticks_config)

        # ticks
        keys = ['labeltop', 'labelright', 'labelcolor', 'direction']
        tick_config = self.filter_config(ax_config, keys, prefix='tick_', index=index)
        if tick_config:
            ax.tick_params(**tick_config)

        # xlim
        xlim_config = self.filter_config(ax_config, ['xlim'], prefix='xlim_', index=index)
        if 'xlim' in xlim_config:
            xlim_config['left'] = xlim_config.get('left', xlim_config.pop('xlim'))
        if xlim_config:
            ax.set_xlim(**xlim_config)

        # ylim
        ylim_config = self.filter_config(ax_config, ['ylim'], prefix='ylim_', index=index)
        if 'ylim' in ylim_config:
            ylim_config['bottom'] = ylim_config.get('bottom', ylim_config.pop('ylim'))
        if ylim_config:
            ax.set_ylim(**ylim_config)

        # colorbar
        if any(to_list(self.config['colorbar'])):
            keys = ['colorbar', 'width', 'pad', 'fake', 'ax_objects']
            colorbar_config = self.filter_config(ax_config, keys, prefix='colorbar_', index=index)
            colorbar_config['ax_image'] = ax_objects['images'][0]
            # if colorbar is disabled for subplot, add param to plot fake axis instead to keep proportions
            colorbar_config['fake'] = not colorbar_config.pop('colorbar', True)
            if 'pad' not in colorbar_config:
                pad = 0.4
                labelright = self.filter_config(ax_config, 'labelright', prefix='tick_', index=index)
                if labelright:
                    ax_x1 = self.get_bbox(ax).x1
                    yticklabels = ax.get_yticklabels()
                    max_ytick_label_x1 = max(self.get_bbox(label).x1
                                             for label in yticklabels[len(yticklabels)//2:])
                    pad += (max_ytick_label_x1 - ax_x1) # account for width of yticklabels to the right of the subplot
                colorbar_config['pad'] = pad
            ax_objects['colorbar'] = self.add_colorbar(**colorbar_config)

        # legend
        keys = ['labels', 'handles', 'size', 'loc', 'ha', 'va', 'handletextpad']
        legend_config = self.filter_config(ax_config, keys, prefix='legend_')
        if legend_config.get('labels') or legend_config.get('handles'):
            if 'cmap' in ax_config:
                colors = self.filter_config(ax_config, 'cmap', index=index)
            if 'color' in ax_config:
                colors = self.filter_config(ax_config, 'color', index=index)
            if 'color' in legend_config:
                colors = legend_config.pop('color')
            legend_config['colors'] = colors
            legend_config['alphas'] = ax_config.get('alpha')
            self.add_legend(ax, mode=mode, **legend_config)

        # grid
        grid = self.filter_config(ax_config, 'grid', index=index)
        grid_keys = ['color', 'linestyle', 'freq']

        minor_config = self.filter_config(ax_config, grid_keys, prefix='minor_grid_', index=index)
        if grid in ('minor', 'both') and minor_config:
            self.add_grid(ax, grid_type='minor', **minor_config)

        major_config = self.filter_config(ax_config, grid_keys, prefix='major_grid_', index=index)
        if grid in ('major', 'both') and minor_config:
            self.add_grid(ax, grid_type='major', **major_config)

        spine_colors = ax_config.get('spine_color')
        if spine_colors is not None:
            spines = ax.spines.values()
            spine_colors = spine_colors if isinstance(spine_colors, list) else [spine_colors] * len(spines)
            for spine, color in zip(spines, spine_colors):
                spine.set_edgecolor(color)

        facecolor = ax_config.get('facecolor', None)
        if facecolor is not None:
            ax.set_facecolor(facecolor)

        ax.set_axisbelow(ax_config.get('set_axisbelow', False))

        if ax_config.get('disable_axes'):
            ax.set_axis_off()
        elif not ax.axison:
            ax.set_axis_on()

        return ax_objects, ax_config

    def fig_annotate(self):
        """ Put suptitle with given parameters over figure and apply `tight_layout`. """
        fig_objects = {}

        text_keys = ['size', 'family', 'color']
        text_config = self.filter_config(self.config, text_keys, prefix='text_')

        # suptitle
        keys = ['suptitle', 't', 'y']
        suptitle_config = self.filter_config(self.config, keys, prefix='suptitle_')
        t = None
        if 'label' in suptitle_config:
            t = suptitle_config.pop('label')
        if 'suptitle' in suptitle_config:
            t = suptitle_config.pop('suptitle')
        if 't' in suptitle_config:
            t = suptitle_config.pop('t')
        suptitle_config['t'] = t
        suptitle_config = {**text_config, **suptitle_config}
        if suptitle_config:
            fig_objects['suptitle'] = fig_objects['suptitle'] = self.fig.suptitle(**suptitle_config)

        self.fig.tight_layout()

        return fig_objects

    def show(self):
        """ Display figure with either `IPython.display` or (if not available) with `plt.show`. """
        try:
            from IPython.display import display #pylint: disable=import-outside-toplevel
            display(self.fig)
        except ImportError:
            plt.show()

    def save(self, **kwargs):
        """ Save plot. """
        default_config = {
            'savepath': datetime.now().strftime('%Y-%m-%d_%H:%M:%S.png'),
            'bbox_inches': 'tight',
            'pad_inches': 0,
            'dpi': 100
        }

        save_keys = ['savepath', 'bbox_inches', 'pad_inches', 'dpi']
        save_config = self.filter_config(kwargs, save_keys, prefix='save_')
        save_config = {**default_config, **save_config}
        savepath = save_config.pop('savepath')

        self.fig.savefig(fname=savepath, **save_config)


    # Rendering methods
    MASK_COLORS = ['firebrick', 'mediumseagreen', 'thistle', 'darkorange', 'navy', 'gold',
                   'red', 'turquoise', 'darkorchid', 'darkkhaki', 'royalblue', 'yellow',
                   'chocolate', 'forestgreen', 'lightpink', 'darkslategray', 'deepskyblue', 'wheat']

    IMSHOW_DEFAULTS = {
        # image
        'cmap': 'Greys_r',
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
        'minor_grid_x_n': 2,
        'minor_grid_y_n': 2,
    }

    @classmethod
    def ax_imshow(cls, data, ax, config, idx_fix=None):
        """ TODO """
        images = []

        mask_colors_generator = cycle(cls.MASK_COLORS)

        for layer_idx, image in enumerate(data):
            if idx_fix is not None:
                layer_idx += idx_fix

            imshow_keys = ['vmin', 'vmax', 'interpolation', 'alpha', 'extent', 'order_axes', 'mask_values']
            imshow_config = cls.filter_config(config, imshow_keys, prefix='imshow_', index=layer_idx)

            cmap = cls.filter_config(config, 'cmap', index=layer_idx)
            # Add `0` to a list of values that shouldn't be displayed if image is a binary mask
            if tuple(np.unique(image)) in [(0, ), (0, 1)]:
                imshow_config['mask_values'] = 0
                imshow_config['vmin'] = 0
                if not is_color_like(cmap):
                    cmap = next(mask_colors_generator)

            # Assemble colormap from given parameters
            # If a single color provided, prepend 'white' color, so that a resulting list defines binary colormap
            if is_color_like(cmap):
                cmap = ['white', cmap]
            # If a list of colors provided in `cmap` argument convert it into a colormap
            if isinstance(cmap, list):
                cmap = cls.make_cmap(colors=cmap)
            else:
                cmap = copy(plt.get_cmap(cmap))
            # Set a color for nan/masked values display to colormap if provided
            mask_color = cls.filter_config(config, 'mask_color', index=layer_idx)
            cmap.set_bad(color=mask_color)
            # Add created cmap to imshow config
            imshow_config['cmap'] = cmap

            # Use a proxy for imshow calls that fixes data preprocessing parameters
            # and re-applies them to axes image before `set_data` calls
            image = preprocess_and_imshow(ax=ax, array=image, **imshow_config)
            images.append(image)

        return {'images': images}, config


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

    @classmethod
    def ax_hist(cls, data, ax, config, idx_fix=None):
        """ TODO """
        bars = []

        for layer_idx, array in enumerate(data):
            if idx_fix is not None:
                layer_idx += idx_fix

            hist_keys = ['bins', 'color', 'alpha', 'label']
            hist_config = cls.filter_config(config, hist_keys, prefix='hist_', index=layer_idx)

            mask_values = cls.filter_config(config, 'mask_values', index=layer_idx)
            if mask_values is None:
                mask_values = []
            else:
                mask_values = to_list(mask_values)

            masks = [array == m if isinstance(m, Number) else m(array) for m in mask_values]
            mask = reduce(np.logical_or, masks, np.isnan(array))
            new_array = np.ma.array(array, mask=mask).flatten()

            _, _, bar = ax.hist(new_array, **hist_config)
            bars.append(bar)

        return {'bars': bars}, config


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

    @classmethod
    def ax_curve(cls, data, ax, config, idx_fix=None):
        """ TODO """
        lines = []

        for layer_idx, arrays in enumerate(data):
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

            if idx_fix is not None:
                layer_idx += idx_fix

            curve_keys = ['color', 'linestyle', 'alpha', 'label']
            curve_config = cls.filter_config(config, curve_keys, prefix='curve_', index=layer_idx)
            line = ax.plot(x, y, **curve_config)
            lines.extend(line)

            # Change scale of axis, if needed
            if config.get('log'):
                ax.set_yscale('log')

        return {'lines': lines}, config


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

    @classmethod
    def ax_loss(cls, data, ax, config, idx_fix=None):
        """ TODO """
        lines = []

        lr_ax = None
        for layer_idx, arrays in enumerate(data):
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

            if idx_fix is not None:
                layer_idx += idx_fix

            label = cls.filter_config(config, 'label') or f'loss {layer_idx + 1}'
            loss_label = label + f' ⟶ {loss[-1]:2.3f}'
            final_window = cls.filter_config(config, 'final_window')
            if final_window is not None:
                final = np.mean(loss[-final_window:]) #pylint: disable=invalid-unary-operand-type
                loss_label += f"\nmean over last {final_window} iterations={final:2.3f}"

            curve_keys = ['color', 'linestyle', 'linewidth', 'alpha']
            loss_config = cls.filter_config(config, curve_keys, prefix='curve_', index=layer_idx)
            loss_line = ax.plot(loss, label=loss_label, **loss_config)
            lines.extend(loss_line)

            window = cls.filter_config(config, 'window', index=layer_idx)
            if window:
                averaged = convolve(loss, np.ones(window), mode='nearest') / window
                mean_color = cls.scale_lightness(loss_config['color'], scale=.5)
                averaged_loss_label = label + ' running mean'
                average_line = ax.plot(averaged, label=averaged_loss_label, color=mean_color, linestyle='--')
                lines.extend(average_line)

            # Change scale of axis, if needed
            if config.get('log_loss'):
                ax.set_yscale('log')

            if lr is not None:
                if lr_ax is None:
                    lr_ax = ax.twinx()
                lr_label = f'learning rate №{layer_idx + 1} ⟶ {lr[-1]:.0e}'
                lr_config = cls.filter_config(config, curve_keys, prefix='lr_', index=layer_idx)
                lr_line = lr_ax.plot(lr, label=lr_label, **lr_config)
                lr_ax.set_ylabel('Learning rate', fontsize=12)
                lines.extend(lr_line)

            if lr is not None and config.get('log_lr'):
                lr_ax.set_yscale('log')

        return {'lines': lines}, config

    # Supplementary methods

    @staticmethod
    def filter_config(config, keys=None, prefix='', index=None):
        """ Make a subdictionary of parameters with required keys.

        Parameter are retrieved if:
        a. It is explicitly requested (via `keys` arg).
        b. Its name starts with given prefix (defined by `prefix` arg).

        Parameters
        ----------
        config : dict
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
        """
        # get value by index if it is requested and value is a list
        maybe_index = lambda value, index: value[index] if index is not None and isinstance(value, list) else value

        if isinstance(keys, str):
            value = config.get(keys, None)
            return maybe_index(value, index)

        if keys is None:
            keys = list(config.keys())
        elif prefix:
            keys += [key.split(prefix)[1] for key in config if key.startswith(prefix)]

        result = {}

        for key in keys:
            if prefix + key in config:
                value = config[prefix + key]
            elif key in config:
                value = config[key]
            else:
                continue
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

    def add_colorbar(self, ax_image, width=.2, pad=None, color='black', fake=False):
        """ Append colorbar to the image on the right. """
        divider = axes_grid1.make_axes_locatable(ax_image.axes)
        cax = divider.append_axes("right", size=width, pad=pad)

        if fake:
            cax.set_axis_off()
            colorbar = None
        else:
            colorbar = ax_image.axes.figure.colorbar(ax_image, cax=cax)
            colorbar.ax.yaxis.set_tick_params(color=color, labelcolor=color)

        return colorbar

    def add_legend(self, ax=None, mode='imshow', handles=None, labels=None, colors='none',
                   alphas=1, size=10, ha=None, va=None, handletextpad=None, **kwargs):
        """ Add patches to axes legend.

        Parameters
        ----------
        ax : int or instance of `matploblib.axes.Axes`
            Axes to put labels into. If and int, used for indexing `self.axes`.
        mode : 'imshow', 'hist', 'curve', 'loss', 'wiggle'
            Mode to match legend hadles patches to.
            If from ('imshow', 'hist', 'wiggle'), use rectangular legend patches.
            If from ('curve', 'loss'), use line legend patches.
        handles : None or sequence of `matplotlib.artist.Artist`
            A list of Artists (lines, patches) to be added to the legend.
            The length of handles and labels (if both provided) should be the same.
        labels : str or list of str
            A list of labels to show next to the artists.
        colors : valid matplotlib color or a list of them
            Color to use for patches creation if those are not provided explicitly.
        alphas : number or list of numbers from 0 to 1
            Legend handles opacity.
        size : int
            Legend size.
        kwargs : misc
            For `matplotlib.legend`.
        """
        if handles is None and labels is None:
            raise ValueError("Both `handle` and `label` cannot be None.")

        if isinstance(ax, int):
            ax = self.axes[ax]

        # get legend that already exists
        legend = ax.get_legend()
        old_handles = getattr(legend, 'legendHandles', [])
        old_handles = [handle.cmap.name if isinstance(handle, PatchCollection) else handle for handle in old_handles]
        texts = getattr(legend, 'texts', [])
        old_labels = [t._text for t in texts] # pylint: disable=protected-access
        handler_map = getattr(legend, '_custom_handler_map', {})

        # form new handles and labels
        new_labels = [] if labels is None else to_list(labels)
        if handles is None:
            colors = colors if isinstance(colors, list) else [colors] * len(new_labels)
            alphas = alphas if isinstance(alphas, list) else [alphas] * len(new_labels)

            new_handles = []
            for color, alpha in zip(colors, alphas):
                if mode in ('imshow', 'hist', 'wiggle'):
                    if is_color_like(color):
                        handle = Patch(color=color, alpha=alpha)
                    else:
                        handle = color
                        handler_map[str] = ColorMappingHandler()
                elif mode in ('curve', 'loss'):
                    handle = Line2D(xdata=[0], ydata=[0], color=color, alpha=alpha)
                new_handles.append(handle)
        else:
            new_handles = to_list(handles)
            new_labels = [handle.get_label() for handle in new_handles] if len(new_labels) == 0 else new_labels

        # extend existing handles and labels with new ones
        if len(new_handles) > 0 and len(new_labels) > 0:
            kwargs['handles'] = old_handles + new_handles
            kwargs['labels'] = old_labels + new_labels

            legend = ax.legend(prop={'size': size}, handletextpad=handletextpad, handler_map=handler_map, **kwargs)

        # adjust texts alignment
        if ha is not None:
            _ = [text.set_ha(ha) for text in legend.get_texts()]
        if va is not None:
            _ = [text.set_ha(va) for text in legend.get_texts()]

    def add_legend_center(self, ax, labels, colors='none', size=30, **kwargs):
        """ Add text labels to axes center and hide patches if color not provided.

        A convenient method for adding text in box (usually on empty ax).

        Parameters
        ----------
        ax : int or instance of `matploblib.axes.Axes`
            Axes to put labels into. If and int, used for indexing `self.axes`.
        labels : str or list of str
            Labels to put into axes.
        colors : valid matplotlib color or a list of them
            If color is 'none', `handletextpad` is automatically set to `-2` effectively eliminating empty left gap.
        size : int
            Legend size.
        kwargs : misc
            For `matplotlib.legend`.
        """
        handletextpad = -2 if colors == 'none' else None
        self.add_legend(ax=ax, colors=colors, labels=labels, loc=10, size=size, handletextpad=handletextpad, **kwargs)

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
