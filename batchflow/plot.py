""" Plot functions. """
from numbers import Number
from copy import copy
import colorsys
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.patches import Patch
from matplotlib.colors import ColorConverter, ListedColormap, is_color_like
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



class plot:
    """ Plotting backend for matplotlib.

    Overall idea
    ------------
    Simply provide data, plot mode and parameters to the `plot` initialization
    and the class takes care of redirecting params to methods they are meant for.

    The logic behind the process is the following:
    1. Convert some provided parameters from 'plotly' to 'matplotlib' naming convention.
    2. Obtain default params for chosen mode and merge them with provided params.
    3. Put data into a double-nested list (via `make_nested_data`).
       Nestedness levels define subplot and layer data order correspondingly.
    4. Parse axes or create them if none provided via `make_or_parse_axes`.
    5. For every axis-data pair:
       a. Filter params relevant for ax (via `filter_dict`).
       b. Call chosen plot method (one of `imshow`, `wiggle`, `hist` or `curve`) with ax params.
       c. Apply all annotations with ax params (via `annotate_axis`).
    6. Show and save figure (via `show_and_save`).

    Data display scenarios
    ----------------------
    1. The simplest one if when one provide a single data array.
    2. A more advanced use case is when one provide a list of arrays:
       a. Images are put on same axis and overlaid: data=[image_1, image_2];
       b. Images are put on separate axes: data=[image_1, image_2], separate=True;
    3. The most complex scenario is when one wish to display images in a 'mixed' manner.
       For example, to overlay first two images but to display the third one separately, one must use:
       data = [[image_1, image_2], image_3];

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
        self.shapes, self.data, self.combine, self.ax_num_shifts = self.parse_data(data=data, combine=combine)
        self.n_subplots = len(self.shapes)
        self.ax_images = np.full(self.n_subplots, None)
        self.ax_configs = np.full(self.n_subplots, None)
        self.fig, self.axes, self.fig_config = self.parse_axes(mode=mode, **kwargs)

        self.plot(mode=mode, **kwargs)

    @staticmethod
    def parse_data(data, combine):
        """ !!. """
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
        for item in data_list:
            if item is None:
                shapes += [(0, 0)]
                data += [None]
            elif isinstance(item, tuple):
                shapes += [item]
                data += [None]
            if isinstance(item, np.ndarray):
                shapes += [item.shape]
                data += [[item]]
            elif isinstance(item, list):
                shapes += [tuple(np.max([subitem.shape for subitem in item], axis=0))]
                data += [item]

        ax_num_shifts = np.cumsum([0] + [item is None for item in data][:-1])

        return shapes, data, combine, ax_num_shifts

    def make_default_config(self, mode, ncols=None, nrows=None, figsize=None, order_axes=None, scale=1,
                            aspect=None, xlim=(None, None), ylim=(None, None), **_):
        """ Infer default figure params from shapes of provided data. """
        config = {'tight_layout': True}

        if mode in ('imshow', 'hist', 'wiggle'):
            col_width = 8 * scale
            default_ncols = 4
        elif mode in ('curve',):
            col_width = 16 * scale
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

        # Make figsize
        if figsize is None and aspect is None:
            if mode == 'imshow':
                if not isinstance(xlim, list):
                    xlim = [xlim] * self.n_subplots
                if not isinstance(ylim, list):
                    ylim = [ylim] * self.n_subplots

                subplots_shapes = []
                order_axes = order_axes or self.IMSHOW_DEFAULTS['order_axes']

                for num, shape in enumerate(self.shapes):
                    min_height = xlim[num][0] or 0
                    max_height = xlim[num][1] or shape[order_axes[1]]
                    subplot_height = abs(max_height - min_height)

                    min_width = ylim[num][0] or shape[order_axes[0]]
                    max_width = ylim[num][1] or 0
                    subplot_width = abs(max_width - min_width)

                    subplots_shapes.append((subplot_width, subplot_height))

                subplots_shapes += [(0, 0)] * (ncols * nrows - len(subplots_shapes))

                heights, widths = np.array(subplots_shapes).reshape((nrows, ncols, 2)).transpose(2, 0, 1)
                max_height, max_width = heights.max(axis=1).sum(), widths.max(axis=0).sum()
                aspect = max_height / max_width
            elif mode == 'hist':
                aspect = 2 / 3 / ncols * nrows
            elif mode == 'wiggle':
                aspect = 1 / ncols * nrows
            elif mode == 'curve':
                aspect = 1 / 3 / ncols * nrows

        if figsize is None:
            fig_width = col_width * ncols
            fig_height = fig_width * aspect
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

            fig, axes = plt.subplots(**config)
            axes = to_list(axes)
            plt.close()
        else:
            axes = to_list(axes)
            fig = axes[0].figure
            config = {}

            if len(axes) < len(self.shapes):
                raise ValueError(f"Not enough axes provided — got ({len(axes)}) for {len(self.shapes)} subplots.")

        return fig, axes, config

    def plot(self, mode, data=None, combine=None, show=False, save=False, **kwargs):
        """ !!. """
        if data is not None:
            self.shapes, self.data, self.combine, self.ax_num_shifts = self.parse_data(data=data, combine=combine)

        mode_defaults = getattr(self, f"{mode.upper()}_DEFAULTS")
        self.config = {**mode_defaults, **kwargs}

        for ax_num, ax in enumerate(self.axes):
            if ax_num >= len(self.data) or self.data[ax_num] is None:
                ax.set_axis_off()
            else:
                filter_index = None if self.combine == 'overlay' else ax_num - self.ax_num_shifts[ax_num]
                self.ax_configs[ax_num] = self.filter_dict(self.config, index=filter_index)
                getattr(self, f"ax_{mode}")(ax_num=ax_num)
                self.ax_annotate(ax_num=ax_num)

        if show:
            self.show()

        if save or 'savepath' in kwargs:
            self.save(kwargs)

        return self

    def __call__(self, mode, **kwargs):
        self.plot(mode=mode, **kwargs)

    def __repr__(self):
        self.show()
        return repr(self.fig)

    def ax_annotate(self, ax_num):
        """ Apply requested annotation functions to given axis with chosen parameters. """
        # pylint: disable=too-many-branches
        ax = self.axes[ax_num]
        ax_config = self.ax_configs[ax_num]
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
        params = self.filter_dict(ax_config, keys, prefix='xlabel_', index=ax_num)
        if params:
            ax.set_xlabel(**params)

        # ylabel
        keys = ['ylabel'] + text_keys
        params = self.filter_dict(ax_config, keys, prefix='ylabel_', index=ax_num)
        if params:
            ax.set_ylabel(**params)

        # aspect
        # params = self.filter_dict(ax_config, ['aspect'], prefix='aspect_', index=ax_num)
        # if params:
        #     ax.set_aspect(**params)

        # xticks
        params = self.filter_dict(ax_config, ['xticks'], prefix='xticks_', index=ax_num)
        if 'xticks' in params:
            params['ticks'] = params.get('ticks', params.pop('xticks'))
        if params:
            ax.set_xticks(**params)

        # yticks
        params = self.filter_dict(ax_config, ['yticks'], prefix='yticks_', index=ax_num)
        if 'yticks' in params:
            params['ticks'] = params.get('ticks', params.pop('yticks'))
        if params:
            ax.set_yticks(**params)

        # ticks
        keys = ['labeltop', 'labelright', 'labelcolor', 'direction']
        params = self.filter_dict(ax_config, keys, prefix='tick_', index=ax_num)
        if params:
            ax.tick_params(**params)

        # xlim
        params = self.filter_dict(ax_config, ['xlim'], prefix='xlim_', index=ax_num)
        if 'xlim' in params:
            params['left'] = params.get('left', params.pop('xlim'))
        if params:
            ax.set_xlim(**params)

        # ylim
        params = self.filter_dict(ax_config, ['ylim'], prefix='ylim_', index=ax_num)
        if 'ylim' in params:
            params['bottom'] = params.get('bottom', params.pop('ylim'))
        if params:
            ax.set_ylim(**params)

        # colorbar
        if any(to_list(self.config['colorbar'])):
            keys = ['colorbar', 'size', 'pad', 'fake', 'ax_image']
            params = self.filter_dict(ax_config, keys, prefix='colorbar_', index=ax_num)
            params['ax_image'] = self.ax_images[ax_num]
            # if colorbar is disabled for subplot, add param to plot fake axis instead to keep proportions
            params['fake'] = not params.pop('colorbar', True)
            self.add_colorbar(**params)

        # legend
        keys = ['label', 'size', 'cmap', 'color', 'loc', 'legend']
        params = self.filter_dict(ax_config, keys, prefix='legend_')
        params['color'] = params.pop('cmap', None) or params.get('color')
        params['label'] = params.pop('legend', None) or params.get('label')
        if params.get('label') is not None:
            self.add_legend(ax, **params)

        # grid
        keys = ['grid', 'b', 'which', 'axis']
        params = self.filter_dict(ax_config, keys, prefix='grid_', index=ax_num)
        params['b'] = params.pop('grid', params.pop('b', 'False'))
        if params:
            ax.grid(**params)

        if ax_config.get('facecolor'):
            ax.set_facecolor(ax_config['facecolor'])

        ax.set_axisbelow(ax_config.get('set_axisbelow', False))

        if ax_config.get('disable_axes'):
            ax.set_axis_off()
        elif not ax.axison:
            ax.set_axis_on()

    def show(self):
        return plt.figure(self.fig)

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
        'facecolor': 'white',
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
        # other
        'order_axes': (1, 0, 2),
        'bad_color': (.0,.0,.0,.0),
        'bad_values': (),
        'transparize_masks': None,
    }

    def ax_imshow(self, ax_num, **kwargs):
        """ !!. """
        ax_config = self.ax_configs[ax_num]

        for image_num, image in enumerate(self.data[ax_num]):
            filter_index = image_num + ax_num - self.ax_num_shifts[ax_num] if self.combine == 'separate' else image_num

            other_keys = ['order_axes', 'bad_values', 'bad_color', 'transparize_masks']
            other_params = self.filter_dict(ax_config, other_keys, prefix='imshow_', index=filter_index)

            imshow_keys = ['cmap', 'vmin', 'vmax', 'interpolation', 'alpha', 'extent']
            imshow_params = self.filter_dict(ax_config, imshow_keys, prefix='imshow_', index=filter_index)

            # If a color provided convert it into a colormap, also add color for nan values
            imshow_params['cmap'] = self.make_cmap(imshow_params['cmap'], other_params['bad_color'])

            # Transpose image if order axes is other than (0, 1, 2)
            order_axes = other_params['order_axes'][:image.ndim]
            image = np.transpose(image, axes=order_axes).astype(np.float32)

            # !!.
            imshow_params['extent'] = imshow_params.get('extent') or [0, image.shape[1], image.shape[0], 0]

            # Fill some values with nans to display them with `bad_color`
            if other_params.get('transparize_masks', image_num > 0):
                if tuple(np.unique(image)) in [(0, ), (0, 1)]:
                    imshow_params['vmin'] = imshow_params.get('vmin', 0)
                    other_params['bad_values'] = [0]

            for bad_value in other_params['bad_values']:
                image[image == bad_value] = np.nan

            ax_image = self.axes[ax_num].imshow(image, **imshow_params)
            if image_num == 0:
                self.ax_images[ax_num] = ax_image

        return kwargs


    HIST_DEFAULTS = {
        # hist
        'bins': 50,
        'color': LoopedList(MASK_COLORS),
        'alpha': 0.8,
        'facecolor': 'white',
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

    def ax_hist(self, ax_num):
        """ !!. """
        ax_config = self.ax_configs[ax_num]

        for image_num, array in enumerate(self.data[ax_num]):
            array = array.flatten()

            bad_values = self.filter_dict(ax_config, ['bad_values'], index=image_num)
            for bad_value in bad_values.get('bad_values', []):
                array = array[array != bad_value]

            filter_index = image_num + ax_num - self.ax_num_shifts[ax_num] if self.combine == 'separate' else image_num
            hist_params = self.filter_dict(ax_config, ['bins', 'color', 'alpha'], prefix='hist_', index=filter_index)
            self.axes[ax_num].hist(array, **hist_params)



    CURVE_COLORS = ['skyblue', 'sandybrown', 'lightpink', 'mediumseagreen', 'thistle', 'firebrick',
                    'forestgreen', 'navy', 'gold', 'red', 'turquoise', 'darkorchid',
                    'darkkhaki', 'royalblue', 'yellow', 'chocolate', 'darkslategray', 'wheat']

    CURVE_DEFAULTS = {
        # main
        'rolling_mean': None,
        'rolling_final': None,
        # curve
        'color': LoopedList(CURVE_COLORS),
        'facecolor': 'white',
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

    def ax_curve(self, ax_num):
        """ !!. """
        ax = self.axes[ax_num]
        ax_config = self.ax_configs[ax_num]

        for image_num, array in enumerate(self.data[ax_num]):
            filter_index = image_num + ax_num - self.ax_num_shifts[ax_num] if self.combine == 'separate' else image_num

            curve_keys = ['color', 'linestyle', 'alpha']
            curve_params = self.filter_dict(ax_config, curve_keys, prefix='curve_', index=filter_index)
            ax.plot(array, **curve_params)

            mean_color = self.scale_lightness(curve_params['color'], scale=.5)

            rolling_mean = ax_config.get('rolling_mean')
            if rolling_mean:
                averaged = array.copy()
                window = min(10 if rolling_mean is True else rolling_mean, len(array))
                if window > len(averaged * 2):
                    break
                averaged[(window // 2):(-window // 2 + 1)] = np.convolve(array, np.ones(window) / window, mode='valid')
                ax.plot(averaged, color=mean_color, linestyle='--')

            final_mean = ax_config.get('final_mean')
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

                ax_config['xlim'] = (0, text_x)

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
    def make_cmap(color, bad_color=None):
        """ Make listed colormap from 'white' and provided color. """
        try:
            cmap = copy(plt.get_cmap(color))
        except ValueError: # if not a valid cmap name, expect it to be a matplotlib color
            if isinstance(color, str):
                color = ColorConverter().to_rgb(color)
            cmap = ListedColormap([(1, 1, 1, 1), color])

        if bad_color is not None:
            cmap.set_bad(color=bad_color)
        return cmap

    @staticmethod
    def scale_lightness(color, scale):
        """ Make new color with modified lightness from existing. """
        if isinstance(color, str):
            color = ColorConverter.to_rgb(color)
        hue, light, saturation = colorsys.rgb_to_hls(*color)
        return colorsys.hls_to_rgb(h=hue, l=min(1, light * scale), s=saturation)

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
    def add_legend(ax, color, label, size, loc, facecolor='white'):
        """ Add patches to legend. All invalid colors are filtered. """
        handles = getattr(ax.get_legend(), 'legendHandles', [])
        colors = [color for color in to_list(color) if is_color_like(color)]
        labels = to_list(label)
        new_patches = [Patch(color=color, label=label) for color, label in zip(colors, labels) if label]
        handles += new_patches
        if handles:
            ax.legend(handles=handles, loc=loc, prop={'size': size}, facecolor=facecolor)

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
