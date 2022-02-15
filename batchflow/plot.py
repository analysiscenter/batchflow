""" Plot functions. """
from colorsys import rgb_to_hls, hls_to_rgb
from copy import copy
from datetime import datetime
from functools import reduce
from numbers import Number

import numpy as np

from IPython.display import display
from matplotlib import patheffects
from matplotlib import pyplot as plt
from matplotlib.colors import ColorConverter, ListedColormap, is_color_like
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits import axes_grid1

from .utils import to_list



class LoopedList(list):
    """ List that loops from given position (default is 0).

        Examples
        --------
        >>> l = LoopedList(['a', 'b', 'c'])
        >>> [l[i] for i in range(9)]
        ['a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c']

        >>> l = LoopedList(['a', 'b', 'c', 'd'], loop_from=2)
        >>> [l[i] for i in range(9)]
        ['a', 'b', 'c', 'd', 'c', 'd', 'c', 'd', 'c']

        >>> l = LoopedList(['a', 'b', 'c', 'd', 'e'], loop_from=-1)
        >>> [l[i] for i in range(9)]
        ['a', 'b', 'c', 'd', 'e', 'e', 'e', 'e', 'e']
    """
    def __init__(self, *args, loop_from=0, **kwargs):
        self.loop_from = loop_from
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx):
        if idx >= len(self):
            pos = self.loop_from + len(self) * (self.loop_from < 0)
            if pos < 0:
                raise IndexError(f"List of length {len(self)} is looped from {self.loop_from} index")
            idx = pos + (idx - pos) % (len(self) - pos)
        return super().__getitem__(idx)

class preprocess_and_imshow:
    """ TODO """
    def __init__(self, ax, X, *args, mask_values=(), order_axes=None, vmin=None, vmax=None, **kwargs):
        self.mask_values = to_list(mask_values) if mask_values is not None else []
        self.order_axes = order_axes
        self.vmin, self.vmax = vmin, vmax

        X_new = self._preprocess(X)
        self.im = ax.imshow(X_new, *args, vmin=vmin, vmax=vmax, **kwargs)

    def _preprocess(self, X):
        masks = [X == m if isinstance(m, Number) else m(X) for m in self.mask_values]
        mask = reduce(np.logical_or, masks, np.isnan(X))
        X_new = np.ma.array(X, mask=mask)

        order_axes = self.order_axes[:X.ndim]
        X_new = np.transpose(X_new, order_axes)
        return X_new

    def set_data(self, A):
        """ TODO """
        vmin_new = np.nanmin(A) if self.vmin is None else self.vmin
        vmax_new = np.nanmax(A) if self.vmax is None else self.vmax
        clim = [vmin_new, vmax_new]
        self.im.set_clim(clim)

        A_new = self._preprocess(A)
        self.im.set_data(A_new)

    def __getattr__(self, key):
        if self.im is None:
            return getattr(self, key)
        return getattr(self.im, key)

    def __repr__(self):
        if self.im is None:
            return super().__repr__()
        return self.im.__repr__()


class plot:
    """ Plotting backend for matplotlib.

    Overall idea
    ------------
    Simply provide data, plot mode and parameters to the `plot` initialization
    and the class takes care of redirecting params to methods they are meant for.

    The logic behind the process is the following:
    1. Parse data:
        - Calculate subplots shapes data.
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
    Address docs of `imshow`, `wiggle`, `hist`, `curve` and `annotate_axis` for details.

    This also allows one to pass arguments of the same name for different plotting steps.
    E.g. `plt.set_title` and `plt.set_xlabel` both require `fontsize` argument.
    Providing `{'fontsize': 30}` in kwargs will affect both title and x-axis labels.
    To change parameter for title only, one can provide {'title_fontsize': 30}` instead.
    """
    def __init__(self, data=None, combine='overlay', mode='imshow', **kwargs):
        """ Plot manager.

        Parses axes from kwargs if provided, else creates them.
        Filters parameters and calls chosen plot method for every axis-data pair.

        Parameters
        ----------
        data : np.ndarray or a list of np.ndarray objects (possibly nested)
            If list has level 1 nestedness, 'overlay/separate' logic is handled via `combine` parameter.
            If list has level 2 nestedness, outer level defines subplots order while inner one defines layers order.
            Shape of data items depends on chosen mode (see below).
        mode : 'imshow', 'wiggle', 'hist', 'curve'
            If 'imshow' plot given arrays as images.
            If 'wiggle' plot 1d subarrays of given array as signals.
            Subarrays are extracted from given data with fixed step along vertical axis.
            If 'hist' plot histogram of flattened array.
            If 'curve' plot given arrays as curves.
        combine : 'overlay', 'separate' or 'mixed'
            Whether overlay images on a single axis, show them on separate ones or use mixed approach.
            Note, that 'wiggle' plot mode is incompatible with `combine='separate'`.
        return_figure : bool
            Whether return created figure or not.
        show : bool
            Whether display created figure or not.
        kwargs :
            - For one of `imshow`, 'wiggle`, `hist` or `curve` (depending on chosen mode).
              Parameters and data nestedness levels must match.
              Every param with 'imshow_', 'wiggle_', 'hist_' or 'curve_' prefix is redirected to corresponding method.
            - For `annotate_axis`.
              Every param with 'title_', 'suptitle_', 'xlabel_', 'ylabel_', 'xticks_', 'yticks_', 'xlim_', 'ylim_',
              colorbar_', 'legend_' or 'grid_' prefix is redirected to corresponding matplotlib method.
              Also 'facecolor', 'set_axisbelow', 'disable_axes' arguments are accepted.
        """
        self.shapes, self.data, self.combine, self.rel_idx_corrections = self.parse_data(data=data, combine=combine)
        self.n_subplots = len(self.shapes)
        self.fig, self.axes, self.fig_config = self.parse_axes(mode=mode, **kwargs)
        self.axes_configs = np.full(len(self.axes), None)
        self.axes_objects = np.full(len(self.axes), None)

        self.plot(mode=mode, **kwargs)

    @staticmethod
    def parse_data(data, combine):
        """ TODO """
        if isinstance(data, int):
            data_list = [None] * data
        elif data is None or isinstance(data, (tuple, np.ndarray)):
            data_list = [data]
        elif isinstance(data, list):
            if isinstance(data[0], Number):
                data_list = [np.array(data)]
            if all(isinstance(item, np.ndarray) for item in data):
                data_list = [[item] for item in data] if combine == 'separate' else [data]
            else:
                combine = 'mixed'
                data_list = data

        shapes = []
        data = []
        none = [0]

        for item in data_list:
            if item is None:
                shapes += [(0, 0)]
                data += [None]
                none.append(1)
            elif isinstance(item, tuple):
                shapes += [item]
                data += [None]
                none.append(1)
            if isinstance(item, np.ndarray):
                shapes += [item.shape]
                data += [[item]]
                none.append(0)
            elif isinstance(item, list):
                shapes += [tuple(np.max([subitem.shape for subitem in item], axis=0))]
                data += [item]
                none.append(0)

        return shapes, data, combine, np.cumsum(none)[:-1]

    def make_default_config(self, mode, ncols=None, nrows=None, figsize=None, scale=1, ratio=None,
                            xlim=(None, None), ylim=(None, None), min_fig_width=4, min_fig_height=4,
                            max_fig_width=25, max_fig_height=15, **kwargs):
        """ Infer default figure params from shapes of provided data. """
        config = {'tight_layout': True, 'facecolor': 'snow'}

        if mode in ('imshow', 'hist', 'wiggle'):
            default_ncols = 4
        elif mode in ('curve',):
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

        if mode in ('imshow', 'hist', 'wiggle'):
            fig_width = 8 * ncols * scale
        elif mode in ('curve',):
            fig_width = 16 * ncols * scale

        # Make figsize
        if figsize is None and ratio is None:
            if mode == 'imshow':
                if not isinstance(xlim, list):
                    xlim = [xlim] * self.n_subplots
                if not isinstance(ylim, list):
                    ylim = [ylim] * self.n_subplots

                widths = []
                heights = []
                for idx, shape in enumerate(self.shapes):
                    order_axes = self.filter_dict(kwargs, 'order_axes', index=idx)
                    order_axes = order_axes or self.IMSHOW_DEFAULTS['order_axes']

                    min_height = xlim[idx][0] or 0
                    max_height = xlim[idx][1] or shape[order_axes[0]]
                    subplot_height = abs(max_height - min_height)
                    heights.append(subplot_height)

                    min_width = ylim[idx][0] or shape[order_axes[1]]
                    max_width = ylim[idx][1] or 0
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

            elif mode == 'curve':
                ratio = 1 / 3 / ncols * nrows

        if figsize is None:
            fig_height = fig_width * ratio

            if fig_height > max_fig_height:
                fig_height = max_fig_height
                fig_width = fig_height / ratio

            if fig_width > max_fig_width:
                fig_width = max_fig_width
                fig_height = fig_width * ratio

            if fig_height < min_fig_height:
                fig_height = min_fig_height
                fig_width = fig_height / ratio

            if fig_width < min_fig_width:
                fig_width = min_fig_width
                fig_height = fig_width * ratio

            figsize = (fig_width, fig_height)

        config['figsize'] = figsize

        return config

    def parse_axes(self, mode, axes=None, axis=None, ax=None, **kwargs):
        """ Create figure and axes if needed. """
        axes = axes or axis or ax

        if axes is None:
            config = self.make_default_config(mode=mode, **kwargs)
            subplots_keys = ['figsize', 'facecolor', 'dpi', 'ncols', 'nrows', 'tight_layout', 'gridspec_kw']
            config = self.filter_dict(kwargs, subplots_keys, prefix='figure_', save_to=config)

            with plt.ioff():
                fig, axes = plt.subplots(**config)
            axes = to_list(axes)
        else:
            axes = to_list(axes)
            fig = axes[0].figure
            config = {}

            if len(axes) < len(self.shapes):
                raise ValueError(f"Not enough axes provided — got ({len(axes)}) for {len(self.shapes)} subplots.")

        return fig, axes, config

    def plot(self, data=None, mode='imshow', combine=None, save=False, axes_idx=None, **kwargs):
        """ TODO

        Notes to self: Explain `abs_idx` and `rel_idx`.
        """
        if data is not None:
            self.shapes, self.data, self.combine, self.rel_idx_corrections = self.parse_data(data=data, combine=combine)

        mode_defaults = getattr(self, f"{mode.upper()}_DEFAULTS")
        self.config = {**mode_defaults, **kwargs}

        axes_indices = range(len(self.axes)) if axes_idx is None else to_list(axes_idx)
        for rel_idx, abs_idx in enumerate(axes_indices):
            ax = self.axes[abs_idx]
            if rel_idx >= len(self.data) or self.data[rel_idx] is None:
                ax.set_axis_off()
            else:
                plot_method = getattr(self, f"ax_{mode}")
                axes_objects, ax_config = plot_method(ax=ax, idx=rel_idx)
                self.ax_annotate(ax=ax, ax_config=ax_config, ax_image=axes_objects[0], idx=rel_idx, mode=mode)

                self.axes_objects[abs_idx] = axes_objects
                self.axes_configs[abs_idx] = ax_config

        if save or 'savepath' in kwargs:
            self.save(kwargs)

        return self

    def __call__(self, mode, **kwargs):
        self.plot(mode=mode, **kwargs)

    def _ipython_display_(self):
        self.show()

    def __repr__(self):
        return repr(self.fig).replace('Figure', 'Batchflow Figure')

    def ax_annotate(self, ax, ax_config, ax_image, idx, mode):
        """ Apply requested annotation functions to given axis with chosen parameters. """
        # pylint: disable=too-many-branches
        text_keys = ['fontsize', 'family', 'color']

        # title
        keys = ['title', 'y'] + text_keys
        params = self.filter_dict(ax_config, keys, prefix='title_')
        params['label'] = params.pop('title', params.pop('label', None))
        if params:
            ax.set_title(**params)

        # suptitle
        keys = ['suptitle', 't', 'y'] + text_keys
        params = self.filter_dict(ax_config, keys, prefix='suptitle_')
        params['t'] = params.pop('t', params.pop('suptitle', params.pop('label', None)))
        if params:
            ax.figure.suptitle(**params)

        # xlabel
        keys = ['xlabel'] + text_keys
        params = self.filter_dict(ax_config, keys, prefix='xlabel_', index=idx)
        if params:
            ax.set_xlabel(**params)

        # ylabel
        keys = ['ylabel'] + text_keys
        params = self.filter_dict(ax_config, keys, prefix='ylabel_', index=idx)
        if params:
            ax.set_ylabel(**params)

        # xticks
        params = self.filter_dict(ax_config, ['xticks'], prefix='xticks_', index=idx)
        if 'xticks' in params:
            params['ticks'] = params.get('ticks', params.pop('xticks'))
        if params:
            ax.set_xticks(**params)

        # yticks
        params = self.filter_dict(ax_config, ['yticks'], prefix='yticks_', index=idx)
        if 'yticks' in params:
            params['ticks'] = params.get('ticks', params.pop('yticks'))
        if params:
            ax.set_yticks(**params)

        # ticks
        keys = ['labeltop', 'labelright', 'labelcolor', 'direction']
        params = self.filter_dict(ax_config, keys, prefix='tick_', index=idx)
        if params:
            ax.tick_params(**params)

        # xlim
        params = self.filter_dict(ax_config, ['xlim'], prefix='xlim_', index=idx)
        if 'xlim' in params:
            params['left'] = params.get('left', params.pop('xlim'))
        if params:
            ax.set_xlim(**params)

        # ylim
        params = self.filter_dict(ax_config, ['ylim'], prefix='ylim_', index=idx)
        if 'ylim' in params:
            params['bottom'] = params.get('bottom', params.pop('ylim'))
        if params:
            ax.set_ylim(**params)

        # colorbar
        if any(to_list(self.config['colorbar'])):
            keys = ['colorbar', 'size', 'pad', 'fake', 'ax_image']
            params = self.filter_dict(ax_config, keys, prefix='colorbar_', index=idx)
            params['ax_image'] = ax_image
            # if colorbar is disabled for subplot, add param to plot fake axis instead to keep proportions
            params['fake'] = not params.pop('colorbar', True)
            self.add_colorbar(**params)

        # legend
        keys = ['label', 'size', 'cmap', 'color', 'loc', 'legend', 'ha', 'va']
        params = self.filter_dict(ax_config, keys, prefix='legend_')
        params['color'] = params.pop('color', params.pop('cmap', None))
        params['label'] = params.pop('legend', None) or params.get('label')
        if params.get('label') is not None:
            self.add_legend(ax, mode=mode, **params)

        # grid
        keys = ['grid', 'b', 'which', 'axis']
        params = self.filter_dict(ax_config, keys, prefix='grid_', index=idx)
        params['b'] = params.pop('grid', params.pop('b', 'False'))
        if params:
            ax.grid(**params)

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
        save_params = self.filter_dict(kwargs, save_keys, prefix='save_')
        save_params = {**default_params, **save_params}
        savepath = save_params.pop('savepath')

        self.fig.savefig(fname=savepath, **save_params)


    # Rendering methods
    MASK_COLORS = ['firebrick', 'mediumseagreen', 'thistle', 'darkorange', 'navy', 'gold',
                   'red', 'turquoise', 'darkorchid', 'darkkhaki', 'royalblue', 'yellow',
                   'chocolate', 'forestgreen', 'lightpink', 'darkslategray', 'deepskyblue', 'wheat']

    IMSHOW_DEFAULTS = {
        # image
        'cmap': LoopedList(['Greys_r', *MASK_COLORS], loop_from=1),
        'facecolor': 'snow',
        # suptitle
        'suptitle_color': 'k',
        # title
        'title_color' : 'k',
        # axis labels
        'xlabel': '', 'ylabel': '',
        # colorbar
        'colorbar': False,
        'colorbar_size': 5,
        'colorbar_pad': None,
        # ticks
        'labeltop': True,
        'labelright': True,
        'direction': 'inout',
        # legend
        'legend_loc': 0,
        'legend_size': 10,
        'legend_label': None,
        # common
        'fontsize': 20,
        # grid
        'grid': False,
        # axes order
        'order_axes': (0, 1, 2),
        'transpose': False,
        # values masking
        'mask_color': (0, 0, 0, 0),
    }

    def ax_imshow(self, ax, idx):
        """ TODO """
        subplot_idx = None if self.combine == 'overlay' else idx - self.rel_idx_corrections[idx]
        config = self.filter_dict(self.config, index=subplot_idx)
        images = []

        for image_idx, image in enumerate(self.data[idx]):

            layer_idx = image_idx
            if self.combine == 'separate':
                layer_idx += idx - self.rel_idx_corrections[idx]

            imshow_keys = ['vmin', 'vmax', 'interpolation', 'alpha', 'extent', 'order_axes', 'mask_values']
            imshow_params = self.filter_dict(config, imshow_keys, prefix='imshow_', index=layer_idx)

            # Assemble colormap from given `cmap` and `mask_color` arguments
            cmap = self.filter_dict(config, 'cmap', index=layer_idx)
            # If a single color provided, prepend 'white' color, so that a resulting list defines binary colormap
            if is_color_like(cmap):
                cmap = ['white', cmap]
            # If a list of colors provided in `cmap` argument convert it into a colormap
            if isinstance(cmap, list):
                cmap = self.make_cmap(colors=cmap)
            else:
                cmap = copy(plt.get_cmap(cmap))
            # Set a color for nan/masked values display to colormap if provided
            mask_color = self.filter_dict(config, 'mask_color', index=layer_idx)
            cmap.set_bad(color=mask_color)
            # Add created cmap to a dict of imshow params
            imshow_params['cmap'] = cmap

            # Add `0` to a list of values that shouldn't be displayed if image is a binary mask
            if tuple(np.unique(image)) in [(0, ), (0, 1)]:
                imshow_params['mask_values'] = 0
                imshow_params['vmin'] = 0

            # Use a proxy for imshow calls that fixes data preprocessing parameters
            # and re-applies them to axes image before `set_data` calls
            image = preprocess_and_imshow(ax=ax, X=image, **imshow_params)
            images.append(image)

        return images, config


    HIST_DEFAULTS = {
        # hist
        'bins': 50,
        'color': LoopedList(MASK_COLORS),
        'alpha': 0.8,
        'facecolor': 'snow',
        # suptitle
        'suptitle_color': 'k',
        # title
        'title_color' : 'k',
        # axis labels
        'xlabel': 'Values', 'ylabel': 'Counts',
        'xlabel_color' : 'k', 'ylabel_color' : 'k',
        # legend
        'legend_size': 10,
        'legend_label': None,
        'legend_loc': 0,
        # grid
        'grid': True,
        # common
        'set_axisbelow': True,
        'fontsize': 20,
        'colorbar': False
    }

    def ax_hist(self, ax, idx):
        """ TODO """
        subplot_idx = None if self.combine == 'overlay' else idx - self.rel_idx_corrections[idx]
        config = self.filter_dict(self.config, index=subplot_idx)
        objects = []

        for array_idx, array in enumerate(self.data[idx]):
            layer_idx = array_idx
            if self.combine == 'separate':
                layer_idx += idx - self.rel_idx_corrections[idx]

            hist_params = self.filter_dict(config, ['bins', 'color', 'alpha'], prefix='hist_', index=layer_idx)

            mask_values = self.filter_dict(config, 'mask_values', index=layer_idx) or []
            masks = [array == m if isinstance(m, Number) else m(array) for m in mask_values]
            mask = reduce(np.logical_or, masks, np.isnan(array))
            array_new = np.ma.array(array, mask=mask).flatten()

            obj = ax.hist(array_new, **hist_params)
            objects.append(obj)

        return objects, config



    CURVE_COLORS = ['skyblue', 'sandybrown', 'lightpink', 'mediumseagreen', 'thistle', 'firebrick',
                    'forestgreen', 'navy', 'gold', 'red', 'turquoise', 'darkorchid',
                    'darkkhaki', 'royalblue', 'yellow', 'chocolate', 'darkslategray', 'wheat']

    CURVE_DEFAULTS = {
        # main
        'rolling_mean': None,
        'rolling_final': None,
        # curve
        'color': LoopedList(CURVE_COLORS),
        'facecolor': 'snow',
        # suptitle
        'suptitle_color': 'k',
        # title
        'title_color': 'k',
        # axis labels
        'xlabel': 'x', 'ylabel': 'y',
        'xlabel_color': 'k', 'ylabel_color': 'k',
        # legend
        'legend_loc': 0,
        'legend_size': 10,
        'legend_label': None,
        # common
        'fontsize': 20,
        'grid': True,
        'colorbar': False
    }

    def ax_curve(self, ax, idx):
        """ TODO """
        subplot_idx = None if self.combine == 'overlay' else idx - self.rel_idx_corrections[idx]
        config = self.filter_dict(self.config, index=subplot_idx)
        objects = []

        for array_idx, array in enumerate(self.data[idx]):
            layer_idx = array_idx
            if self.combine == 'separate':
                layer_idx += idx - self.rel_idx_corrections[idx]

            curve_keys = ['color', 'linestyle', 'alpha']
            curve_params = self.filter_dict(config, curve_keys, prefix='curve_', index=layer_idx)
            obj = ax.plot(array, **curve_params)
            objects.append(obj)

            mean_color = self.scale_lightness(curve_params['color'], scale=.5)

            rolling_mean = config.get('rolling_mean')
            if rolling_mean:
                averaged = array.copy()
                window = min(10 if rolling_mean is True else rolling_mean, len(array))
                if window > len(averaged * 2):
                    break
                averaged[(window // 2):(-window // 2 + 1)] = np.convolve(array, np.ones(window) / window, mode='valid')
                ax.plot(averaged, color=mean_color, linestyle='--')

            final_mean = config.get('final_mean')
            if final_mean:
                window = 100 if final_mean is True else final_mean
                mean = np.mean(array[-window:])

                line_len = len(array) // 20
                curve_len = len(array)
                line_x = np.arange(line_len) + curve_len
                line_y = [mean] * line_len
                ax.plot(line_x, line_y, linestyle='--', linewidth=1.2, color=mean_color)

                fontsize = 10
                text_x = curve_len + line_len
                text_y = mean - fontsize / 300
                text = ax.text(text_x, text_y, f"{mean:.3f}", fontsize=fontsize)
                text.set_path_effects([patheffects.Stroke(linewidth=3, foreground='white'), patheffects.Normal()])

                config['xlim'] = (0, text_x)

        return objects, config

    # Supplementary methods

    @staticmethod
    def filter_dict(params, keys=None, prefix='', index=None, save_to=None):
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
    def add_colorbar(ax_image, size=5, pad=None, color='black', fake=False):
        """ Append colorbar to the image on the right. """
        divider = axes_grid1.make_axes_locatable(ax_image.axes)
        pad = size * 1.5 if pad is None else pad
        cax = divider.append_axes("right", size=f"{size}%", pad=f"{pad}%")
        if fake:
            cax.set_axis_off()
        else:
            colorbar = ax_image.axes.figure.colorbar(ax_image, cax=cax)
            colorbar.ax.yaxis.set_tick_params(color=color)
            ax_image.axes.created_colorbar = colorbar

    @staticmethod
    def add_legend(ax, mode, color, label, size, loc, facecolor='white', ha=None, va=None, **kwargs):
        """ TODO Add patches to legend. All invalid colors are filtered.

        Notes to self: Rewrite doc, explain line/patches parametrization.
        """
        legend = ax.get_legend()
        handles = getattr(legend, 'legendHandles', [])
        texts = getattr(legend, 'get_texts', lambda: [])()
        labels = [t._text for t in texts]

        colors = [color for color in to_list(color) if is_color_like(color)]
        if mode in ('imshow', 'hist', 'wiggle'):
            patch_config = {
                **kwargs
            }
            handles += [Patch(color=color, **patch_config) for color in colors]
        elif mode in ('curve', ):
            line_config = {
                'xdata': np.array([0, 2]) * size,
                'ydata': np.array([0.35, 0.35]) * size,
                'linewidth': 1.5,
                'linestyle': '-',
                **kwargs
                }
            handles += [Line2D(color=color, **line_config) for color in colors]

        labels += to_list(label)

        legend = ax.legend(handles=handles, labels=labels, loc=loc, prop={'size': size}, facecolor=facecolor)

        if ha is not None:
            _ = [text.set_ha(ha) for text in legend.get_texts()]
        if va is not None:
            _ = [text.set_ha(va) for text in legend.get_texts()]



def plot_loss(data, title=None, **kwargs):
    """ Shorthand for loss plotting. """
    kwargs = {
        'xlabel': 'Iterations',
        'ylabel': 'Loss',
        'label': title or 'Loss graph',
        'xlim': (0, None),
        'rolling_mean': 10,
        'final_mean': 100,
        **kwargs
    }
    return plot(data, mode='curve', backend='matplotlib', **kwargs)
