""" Plot functions. """
from numbers import Number
from copy import copy
import colorsys
from itertools import zip_longest

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.cm import get_cmap, register_cmap
from matplotlib.patches import Patch
from matplotlib.colors import ColorConverter, ListedColormap, LinearSegmentedColormap, is_color_like
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
       a. Filter params relevant for ax (via `filter_parameters`).
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
    def __init__(self, data=None, mode='imshow', separate=False, shapes=0, **kwargs):
        """ Plot manager.

        Parses axes from kwargs if provided, else creates them.
        Filters parameters and calls chosen plot method for every axis-data pair.

        Parameters
        ----------
        data : np.ndarray or a list of np.ndarray objects (possibly nested)
            If list has level 1 nestedness, 'overlaid/separate' logic is handled via `separate` parameter.
            If list has level 2 nestedness, outer level defines subplots order while inner one defines layers order.
            Shape of data items depends on chosen mode (see below).
        mode : 'imshow', 'wiggle', 'hist', 'curve'
            If 'imshow' plot given arrays as images.
            If 'wiggle' plot 1d subarrays of given array as signals.
            Subarrays are extracted from given data with fixed step along vertical axis.
            If 'hist' plot histogram of flattened array.
            If 'curve' plot given arrays as curves.
        separate : bool
            Whether plot images on separate axes instead of putting them all together on a single one.
            Incompatible with 'wiggle' mode.
        shapes : int or tuple or list of tuples
            Defines subplots sizes that needed to be created in addition to those created for `data`.
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
        mode_defaults = getattr(self, f"{mode.upper()}_DEFAULTS")
        all_params = {**mode_defaults, **kwargs}

        self.data = self.make_nested_data(data=data, separate=separate)
        self.axes = self.make_or_parse_axes(mode=mode, shapes=shapes, all_params=all_params)

        self.skips = 0
        for ax_num, (ax_data, ax) in enumerate(zip_longest(self.data, self.axes, fillvalue=None)):
            if ax_data is None:
                ax.set_axis_off()
                self.skips += 1
            else:
                ax_num = ax_num - self.skips
                index_condition = None if separate else lambda x: isinstance(x, list)
                ax_params = self.filter_parameters(all_params, index=ax_num, index_condition=index_condition)
                ax_params = getattr(self, mode)(ax=ax, data=ax_data, **ax_params)
                self.annotate_axis(ax=ax, ax_num=ax_num, ax_params=ax_params, all_params=all_params, mode=mode)

        if 'savepath' in kwargs:
            self.save(kwargs)

    def __repr__(self):
        plt.figure(self.fig)
        return repr(self.fig)

    @property
    def figure(self):
        return self.axes[0].figure

    fig = figure

    # Action methods

    @staticmethod
    def make_nested_data(data, separate):
        """ Construct nested list of data arrays for plotting. """
        if data is None:
            return []
        if isinstance(data, np.ndarray):
            return [[data]]
        if isinstance(data[0], Number):
            return [[np.array(data)]]
        if all(isinstance(item, np.ndarray) for item in data):
            return [[item] for item in data] if separate else [data]
        if separate:
            raise ValueError("Arrays list must be flat, when `separate` option is True.")
        return [[item] if isinstance(item, np.ndarray) else item for item in data]

    def infer_figsize(self, mode, shapes, ncols, nrows,
                      order_axes=None, scale=1, aspect=None, xlim=(None, None), ylim=(None, None)):
        """" Infer figure size from aspect ratios of provided data. """
        MODE_TO_FIGSIZE = {'wiggle' : (12, 7),
                           'curve': (15, 5)}

        DEFAULT_COLUMN_WIDTH = int(8 * scale)
        DEFAULT_ROW_HEIGHT = int(5 * scale)

        data_shapes = [np.max([layer_data.shape for layer_data in subplot_data], axis=0)
                       if subplot_data is not None else (0, 0) for subplot_data in self.data]
        if isinstance(shapes, list):
            data_shapes += shapes
        elif isinstance(shapes, tuple):
            data_shapes += [shapes]
        elif len(self.data) > 0:
            data_shapes += [data_shapes[-1]] * shapes
        else:
            raise ValueError("`shapes` might be an integer only when `data` is provided.")

        if mode == 'imshow':
            if aspect is None:
                if not isinstance(xlim, list):
                    xlim = [xlim] * len(data_shapes)
                if not isinstance(ylim, list):
                    ylim = [ylim] * len(data_shapes)

                subplots_shapes = []
                order_axes = order_axes or self.IMSHOW_DEFAULTS['order_axes']

                for num, shape in enumerate(data_shapes):
                    min_height = xlim[num][0] or 0
                    max_height = xlim[num][1] or shape[order_axes[1]]
                    subplot_height = abs(max_height - min_height)

                    min_width = ylim[num][0] or shape[order_axes[0]]
                    max_width = ylim[num][1] or 0
                    subplot_width = abs(max_width - min_width)

                    subplots_shapes.append((subplot_width, subplot_height))

                subplots_shapes += [(0, 0)] * (ncols * nrows - len(subplots_shapes))

                heights, widths = np.array(subplots_shapes).reshape((nrows, ncols, 2)).transpose(2, 0, 1)
                max_height, max_width = heights.sum(axis=0).max(), widths.sum(axis=1).max()
                aspect = max_height / max_width

            fig_width = min(30, DEFAULT_COLUMN_WIDTH * ncols)
            fig_height = max(DEFAULT_ROW_HEIGHT * nrows, fig_width * aspect)
            figsize = (fig_width, fig_height)

        elif mode == 'hist':
            fig_width = DEFAULT_COLUMN_WIDTH * ncols

            if aspect is None:
                fig_height = DEFAULT_ROW_HEIGHT * nrows
            else:
                fig_height = fig_width / aspect

            figsize = (fig_width, fig_height)

        else:
            width, height = MODE_TO_FIGSIZE[mode]
            width = width * scale

            if aspect is None:
                height *= scale
            else:
                height = width / aspect

            figsize = (width, height)

        return figsize

    @staticmethod
    def infer_cols_rows(n_subplots, ncols=None, nrows=None, **_):
        """ Infer number of columns or/and rows for ploting provided number of subplots. """
        DEFAULT_NCOLS = 4
        ceil_div = lambda a, b: -(-a // b)

        if ncols is None and nrows is None:
            ncols = min(DEFAULT_NCOLS, n_subplots)
            nrows = ceil_div(n_subplots, ncols)
        elif ncols is None:
            ncols = ceil_div(n_subplots, nrows)
        elif nrows is None:
            nrows = ceil_div(n_subplots, ncols)

        return ncols, nrows

    def make_or_parse_axes(self, mode, shapes, all_params):
        """ Create figure and axes if needed, else use provided. """
        axes = all_params.pop('axes', None)
        axes = all_params.pop('axis', axes)
        axes = all_params.pop('ax', axes)

        n_subplots = len(self.data)
        if isinstance(shapes, list):
            n_subplots += len(shapes)
        elif isinstance(shapes, tuple):
            n_subplots += 1
        else:
            n_subplots += shapes

        if axes is None:
            FIGURE_KEYS = ['figsize', 'facecolor', 'dpi', 'ncols', 'nrows', 'tight_layout']
            params = self.filter_parameters(all_params, FIGURE_KEYS, prefix='figure_')

            if n_subplots > 0:
                ncols, nrows = self.infer_cols_rows(n_subplots, **params)
                params['ncols'], params['nrows'] = ncols, nrows

                if 'figsize' not in params:
                    INFER_FIGSIZE_KEYS = ['order_axes', 'scale', 'aspect', 'xlim', 'ylim']
                    infer_figsize_params = self.filter_parameters(all_params, INFER_FIGSIZE_KEYS)
                    params['figsize'] = self.infer_figsize(mode, shapes, ncols, nrows, **infer_figsize_params)

            params['tight_layout'] = params.get('tight_layout', True)

            _, axes = plt.subplots(**params)
            plt.close()

        axes = to_list(axes)
        n_axes = len(axes)
        if n_axes < n_subplots:
            raise ValueError(f"Not enough axes provided ({n_axes}) for {n_subplots} subplots.")

        return axes


    @classmethod
    def annotate_axis(cls, ax, ax_num, ax_params, all_params, mode):
        """ Apply requested annotation functions to given axis with chosen parameters. """
        # pylint: disable=too-many-branches
        TEXT_KEYS = ['fontsize', 'family', 'color']

        # title
        keys = ['title', 'y'] + TEXT_KEYS
        params = cls.filter_parameters(ax_params, keys, prefix='title_', index=ax_num)
        params['label'] = params.pop('title', params.pop('label', None))
        if params:
            ax.set_title(**params)

        # suptitle
        keys = ['suptitle', 't', 'y'] + TEXT_KEYS
        params = cls.filter_parameters(ax_params, keys, prefix='suptitle_')
        params['t'] = params.pop('t', params.pop('suptitle', params.pop('label', None)))
        if params:
            ax.figure.suptitle(**params)

        # xlabel
        keys = ['xlabel'] + TEXT_KEYS
        params = cls.filter_parameters(ax_params, keys, prefix='xlabel_', index=ax_num)
        if params:
            ax.set_xlabel(**params)

        # ylabel
        keys = ['ylabel'] + TEXT_KEYS
        params = cls.filter_parameters(ax_params, keys, prefix='ylabel_', index=ax_num)
        if params:
            ax.set_ylabel(**params)

        # aspect
        # params = cls.filter_parameters(ax_params, ['aspect'], prefix='aspect_', index=ax_num)
        # if params:
        #     ax.set_aspect(**params)

        # xticks
        params = cls.filter_parameters(ax_params, ['xticks'], prefix='xticks_', index=ax_num)
        if 'xticks' in params:
            params['ticks'] = params.get('ticks', params.pop('xticks'))
        if params:
            ax.set_xticks(**params)

        # yticks
        params = cls.filter_parameters(ax_params, ['yticks'], prefix='yticks_', index=ax_num)
        if 'yticks' in params:
            params['ticks'] = params.get('ticks', params.pop('yticks'))
        if params:
            ax.set_yticks(**params)

        # ticks
        keys = ['labeltop', 'labelright', 'labelcolor', 'direction']
        params = cls.filter_parameters(ax_params, keys, prefix='tick_', index=ax_num)
        if params:
            ax.tick_params(**params)

        # xlim
        params = cls.filter_parameters(ax_params, ['xlim'], prefix='xlim_', index=ax_num)
        if 'xlim' in params:
            params['left'] = params.get('left', params.pop('xlim'))
        if params:
            ax.set_xlim(**params)

        # ylim
        params = cls.filter_parameters(ax_params, ['ylim'], prefix='ylim_', index=ax_num)
        if 'ylim' in params:
            params['bottom'] = params.get('bottom', params.pop('ylim'))
        if params:
            ax.set_ylim(**params)

        # colorbar
        if all_params.get('colorbar', False) and mode == 'imshow':
            keys = ['colorbar', 'size', 'pad', 'fake', 'ax_image']
            params = cls.filter_parameters(ax_params, keys, prefix='colorbar_', index=ax_num)
            # if colorbar is disabled for subplot, add param to plot fake axis instead to keep proportions
            params['fake'] = not params.pop('colorbar', True)
            cls.add_colorbar(**params)

        # legend
        keys = ['label', 'size', 'cmap', 'color', 'loc']
        params = cls.filter_parameters(ax_params, keys, prefix='legend_')
        params['color'] = params.pop('cmap', None) or params.get('color')
        if params.get('label') is not None:
            cls.add_legend(ax, **params)

        # grid
        keys = ['grid', 'b', 'which', 'axis']
        params = cls.filter_parameters(ax_params, keys, prefix='grid_', index=ax_num)
        params['b'] = params.pop('grid', params.pop('b', 'False'))
        if params:
            ax.grid(**params)

        if ax_params.get('facecolor'):
            ax.set_facecolor(ax_params['facecolor'])

        ax.set_axisbelow(ax_params.get('set_axisbelow', False))

        if ax_params.get('disable_axes'):
            ax.set_axis_off()
        elif not ax.axison:
            ax.set_axis_on()


    @classmethod
    def save(cls, all_kwargs):
        """ Save plot. """
        default_params = dict(bbox_inches='tight', pad_inches=0, dpi=100)
        SAVE_PARAMS = ['savepath', 'bbox_inches', 'pad_inches', 'dpi']
        params = cls.filter_parameters(all_kwargs, SAVE_PARAMS, prefix='save_')
        params = {**default_params, **params}
        self.fig.savefig(savepath, **params)


    # Rendering methods
    MASK_COLORS = ['firebrick', 'mediumseagreen', 'thistle', 'darkorange', 'navy', 'gold',
                   'red', 'turquoise', 'darkorchid', 'darkkhaki', 'royalblue', 'yellow',
                   'chocolate', 'forestgreen', 'lightpink', 'darkslategray', 'deepskyblue', 'wheat']

    IMSHOW_DEFAULTS = {
        # image
        'cmap': LoopedList(['Greys_r', *MASK_COLORS], loop_from=1),
        'facecolor': 'white',
        # axis labels
        'xlabel': '', 'ylabel': '',
        # colorbar
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
        'transparize_masks': None,
    }

    @classmethod
    def imshow(cls, ax, data, **kwargs):
        """ Plot arrays as images one over another on given axis.

        Parameters
        ----------
        ax : matplotlib axis
            Axis to plot images on.
        data : list of np.ndarray
            Every item must be a valid matplotlib image.
        kwargs :
            order_axes : tuple of ints
                Order of image axes.
            bad_values : list of numbers
                Data values that should be displayed with 'bad_color'.
            transparize_masks : bool, optional
                Whether treat zeros in binary masks as bad values or not.
                If True, make zero values in all binary masks transparent on display.
                If False, do not make zero values in any binary masks transparent on display.
                If not provided, make zero values transparent in all masks that overlay an image.
            params for images drawn by `plt.imshow`:
                - 'cmap', 'vmin', 'vmax', 'interpolation', 'alpha', 'extent'
                - params with 'imshow_' prefix

        Notes
        -----
        See class docs for details on prefixes usage.
        See class and method defaults for arguments examples.
        """
        for image_num, image in enumerate(data):
            image = np.transpose(image, axes=kwargs['order_axes'][:image.ndim]).astype(np.float32)

            keys = ['cmap', 'vmin', 'vmax', 'interpolation', 'alpha', 'extent']
            params = cls.filter_parameters(kwargs, keys, prefix='imshow_', index=image_num)
            params['cmap'] = cls.make_cmap(params.pop('cmap'), kwargs['bad_color'])
            params['extent'] = params.get('extent') or [0, image.shape[1], image.shape[0], 0]

            # fill some values with nans to display them with `bad_color`
            bad_values = cls.filter_parameters(kwargs, ['bad_values'], index=image_num).get('bad_values', [])
            transparize_masks = kwargs.get('transparize_masks')
            transparize_masks = transparize_masks if transparize_masks is not None else image_num > 0
            if transparize_masks:
                unique_values = tuple(np.unique(image))
                if unique_values == (0,) or unique_values == (0, 1): # pylint: disable=consider-using-in
                    params['vmin'] = params.get('vmin', 0)
                    bad_values = [0]
            for bad_value in bad_values:
                image[image == bad_value] = np.nan

            ax_image = ax.imshow(image, **params)
            if image_num == 0:
                kwargs['ax_image'] = ax_image

        return kwargs


    HIST_DEFAULTS = {
        # hist
        'bins': 50,
        'color': LoopedList(MASK_COLORS),
        'alpha': 0.8,
        'facecolor': 'white',
        # suptitle
        'suptitle_color': 'k',
        'suptitle_y': 1.01,
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
        'fontsize': 20
    }

    @classmethod
    def hist(cls, ax, data, **kwargs):
        """ Plot histograms on given axis.

        Parameters
        ----------
        ax : matplotlib axis
            Axis to plot images on.
        data : np.ndarray or list of np.ndarray
            Arrays to build histograms. Can be of any shape since they are flattened.
        kwargs :
            params for overlaid histograms drawn by `plt.hist`:
                - 'bins', 'color', 'alpha'
                - params with 'hist_' prefix

        Notes
        -----
        See class docs for details on prefixes usage.
        See class and method defaults for arguments examples.
        """
        for image_num, array in enumerate(data):
            array = array.flatten()

            bad_values = cls.filter_parameters(kwargs, ['bad_values'], index=image_num)
            for bad_value in bad_values.get('bad_values', []):
                array = array[array != bad_value]

            params = cls.filter_parameters(kwargs, ['bins', 'color', 'alpha'], prefix='hist_', index=image_num)
            ax.hist(array, **params)

        return kwargs



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
        'grid': True
    }

    @classmethod
    def curve(cls, ax, data, **kwargs):
        """ Plot curves on given axis.

        Parameters
        ----------
        ax : matplotlib axis
            Axis to plot images on.
        data : np.ndarray or list of np.ndarray
            Arrays to plot. Must be 1d.
        kwargs :
            rolling_mean : int or None
                If int, calculate and display rolling mean with window `rolling_mean` size.
            rolling_final : int or None
                If int, calculate an display mean over last `rolling_final` array elements.
            params for overlaid curves drawn by `plt.plot`:
                - 'color', 'linestyle', 'alpha'
                - params with 'curve_' prefix

        Notes
        -----
        See class docs for details on prefixes usage.
        See class and method defaults for arguments examples.
        """
        for image_num, array in enumerate(data):
            keys = ['color', 'linestyle', 'alpha']
            params = cls.filter_parameters(kwargs, keys, prefix='curve_', index=image_num)
            ax.plot(array, **params)

            mean_color = cls.scale_lightness(params['color'], scale=.5)

            rolling_mean = kwargs.get('rolling_mean')
            if rolling_mean:
                averaged = array.copy()
                window = min(10 if rolling_mean is True else rolling_mean, len(array))
                if window > len(averaged * 2):
                    break
                averaged[(window // 2):(-window // 2 + 1)] = np.convolve(array, np.ones(window) / window, mode='valid')
                ax.plot(averaged, color=mean_color, linestyle='--')

            final_mean = kwargs.get('final_mean')
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

                kwargs['xlim'] = (0, text_x)

        return kwargs

    # Supplementary methods

    @staticmethod
    def filter_parameters(params, keys=None, prefix='', index=None, index_condition=None):
        """ Make a subdictionary of parameters with required keys.

        Parameter are retrieved if:
        a. It is explicitly requested (via `keys` arg).
        b. Its name starts with given prefix (defined by `prefix` arg).

        Parameters
        ----------
        params : dict
            Arguments to filter.
        keys : sequence
            Keys to retrieve.
        prefix : str, optional
            Arguments with keys starting with given prefix will also be retrieved.
            Defaults to `''`, i.e. no prefix used.
        index : int
            Index of argument value to retrieve.
            If none provided, get whole argument value.
            If value is non-indexable, get it without indexing.
        index_condition : callable
            Function that takes indexed argument value and returns a bool specifying whether should it be really indexed.
        """
        result = {}

        keys = keys or list(params.keys())
        if prefix:
            keys += [key.split(prefix)[1] for key in params if key.startswith(prefix)]

        for key in keys:
            value = params.get(prefix + key, params.get(key))
            if value is None:
                continue
            # check if parameter value indexing is requested and possible
            if index is not None and isinstance(value, list):
                # check if there is no index condition or there is one and it is satisfied
                if index_condition is None or index_condition(value[index]):
                    value = value[index]
            result[key] = value
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
        h, l, s = colorsys.rgb_to_hls(*color)
        return colorsys.hls_to_rgb(h, min(1, l * scale), s = s)

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
