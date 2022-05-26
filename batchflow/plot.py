""" Plot functions. """
from ast import literal_eval
from colorsys import rgb_to_hls, hls_to_rgb
from copy import copy
from datetime import datetime
from numbers import Number
import operator

import numpy as np

import cv2
from scipy.ndimage import convolve
from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.colors import ColorConverter, ListedColormap, is_color_like
from matplotlib.patches import Patch, Rectangle
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerBase
from matplotlib.ticker import MaxNLocator, AutoMinorLocator
from mpl_toolkits import axes_grid1
from numba import njit

from .utils import to_list

STR_TO_OPERATION = {
    '<': operator.lt,
    '<=': operator.le,
    '==': operator.eq,
    '!=': operator.ne,
    '>=': operator.ge,
    '>': operator.gt
}

def evaluate_str_comparison(arg0, string):
    """ Find comparison operator in string and apply it against given argument
    and those parsed from the string to the right of the found operator.
    """
    for key in STR_TO_OPERATION:
        if key in string:
            operation = STR_TO_OPERATION[key]
            arg1 = literal_eval(string.split(key)[-1])
            return operation(arg0, arg1)
    raise ValueError("Given string {string} does not contain any of supported operators: {STR_TO_OPERATION}")


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


class ColorMappingHandler(HandlerBase):
    """ Handler transforming empty patches into colored patches corresponding to collection colormap.

    Notes
    -----
    Used as `handler_map` argument for `plt.legend` to make colormap patches.
    """
    def __init__(self, n_segments=8, **kwargs):
        self.n_segments = n_segments
        super().__init__(**kwargs)

    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        """ Create a rectangle consisting of sequential patches with colors of colormap. """
        _ = legend, fontsize
        segment_width = width / self.n_segments
        cmap = orig_handle.cmap

        segments = []
        for segment_index in range(self.n_segments):
            anchor = [xdescent + segment_index * segment_width, ydescent]
            facecolor = cmap(segment_index / self.n_segments)
            segment = Rectangle(anchor, segment_width, height, facecolor=facecolor, transform=trans)
            segments.append(segment)

        label = orig_handle.get_label()
        patch = PatchCollection(segments, match_original=True, edgecolor=None, cmap=cmap.name, label=label)

        return [patch]

@njit
def is_binary(array):
    """ Fast check that array consists of 0 and 1 only. """
    for item in array:
        if item not in (0., 1.):
            return False
    return True


def contains_numbers(iterable):
    """ Check if first iterable item is a number. """
    return isinstance(iterable[0], Number)

def maybe_index(key, value, index):
    """ Get i-th element of parameter if index is provided and parameter value is a list else return it unchanged.

    Parameters
    ----------
    key : str
        Parameter name.
    value : misc
        Parameter value.
    index : None or int
        If not None, a number to use for parameter indexing.

    Raises
    ------
    ValueError
        If parameter is a list but the index is greater than its length.
    """
    if index is not None and isinstance(value, list):
        try:
            return value[index]
        except IndexError as e:
            msg = f"Tried to obtain element #{index} from `{key}={value}`. Either provide parameter value "\
                    f"as a single item (to use the same `{key}` several times) or add more elements to it."
            raise ValueError(msg) from e
    return value

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
    if isinstance(keys, str):
        value = config.get(keys, None)
        return maybe_index(keys, value, index)

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
        result[key] = maybe_index(key, value, index)

    return result

def make_cmap(colors):
    """ Make colormap from provided color/colors list. """
    colors = [ColorConverter().to_rgb(color) if isinstance(color, str) else color for color in to_list(colors)]
    cmap = ListedColormap(colors)
    return cmap

def scale_lightness(color, scale):
    """ Make new color with modified lightness from existing. """
    if isinstance(color, str):
        color = ColorConverter.to_rgb(color)
    hue, light, saturation = rgb_to_hls(*color)
    new_color = hls_to_rgb(h=hue, l=min(1, light * scale), s=saturation)
    return new_color


class Layer:
    """ Implements subplot layer, storing plotted object and redirecting call to corresponding matplotlib methods.
        Also handles data transformations in `image` mode, re-applying it to given data on `set_data` call.
    """
    def __init__(self, subplot, mode, index, config, data):
        self.subplot = subplot
        self.mode = mode
        self.index = index
        self.config = config

        if mode in ('image', 'histogram') and 'mask_values' not in self.config:
            # Add `0` to a list of values that shouldn't be displayed if image is a binary mask
            if is_binary(data.flatten()):
                self.config['mask_values'] = 0
                self.config['vmin'] = 0
                self.config['vmax'] = 1

        preprocessed_data = self.preprocess(data)
        self.object = getattr(self, mode)(preprocessed_data)

    def preprocess(self, data):
        """ Look through layer config for requested data transformations and apply them ."""
        if self.mode == 'image':
            order_axes = self.config.get('order_axes', None)
            if order_axes is not None:
                order_axes = order_axes[:data.ndim]
                data = np.transpose(data, order_axes)

            dilate = self.config.get('dilate', False)
            if dilate:
                if dilate is True:
                    dilation_config = {'kernel': np.ones((3, 1), dtype=np.uint8)}
                elif isinstance(dilate, int):
                    dilation_config = {'iterations': dilate}
                elif isinstance(dilate, tuple):
                    dilation_config = {'kernel': np.ones(dilate, dtype=np.uint8)}

                data = cv2.dilate(data, **dilation_config)

            masking_conditions = self.config.get('mask', None)
            if masking_conditions is not None:
                mask = np.isnan(data)
                masking_conditions = to_list(masking_conditions)
                for condition in masking_conditions:
                    if isinstance(condition, Number):
                        condition_mask = data == condition
                    elif isinstance(condition, str):
                        condition_mask = evaluate_str_comparison(data, condition)
                    elif callable(condition):
                        condition_mask = condition(data)
                    mask = np.logical_or(mask, condition_mask)
                data = np.ma.array(data, mask=mask)

        if self.mode == 'histogram':
            flatten = self.config.get('flatten', False)
            if flatten:
                data = data.flatten()

        if self.mode == 'curve':
            if isinstance(data, np.ndarray) or (isinstance(data, list) and contains_numbers(data)):
                if hasattr(self, 'object'):
                    xdata = self.object.get_xdata()
                else:
                    xdata = range(len(data))
                data = [xdata, data]

        if self.mode == 'loss':
            if isinstance(data, tuple):
                loss, lr = data
            else:
                loss, lr = data, None

            window = self.config.get('window')
            if window is None:
                smoothed = None
            else:
                smoothed = convolve(loss, np.ones(window), mode='nearest') / window

            data = [loss, smoothed, lr]

        return data

    @property
    def ax(self):
        return self.subplot.ax

    @property
    def twin_ax(self):
        return self.subplot.twin_ax

    def update_lims(self):
        """ Recalculate plot limits. """
        self.ax.relim()
        self.ax.autoscale_view()

    def update(self, data):
        """ Preprocess given data and pass it to `set_data`. Does not work in 'histogram' and 'loss' mode. """
        if self.mode == 'image':
            data = self.preprocess(data)
            self.object.set_data(data)

            vmin = self.config.get('vmin', np.nanmin(data))
            vmax = self.config.get('vmax', np.nanmax(data))
            self.object.set_clim([vmin, vmax])

        if self.mode == 'histogram':
            raise NotImplementedError("Updating layer data is not in supported in 'histogram' mode. ")

        if self.mode == 'curve':
            x_data, y_data = self.preprocess(data)
            self.object.set_data(x_data, y_data)
            self.update_lims()

        if self.mode == 'loss':
            raise NotImplementedError("Updating layer data is not in supported in 'loss' mode. ")


    def image(self, data):
        """ Display data as an image. """
        image_keys = ['vmin', 'vmax', 'interpolation', 'alpha', 'extent']
        image_config = filter_config(self.config, image_keys, prefix='image_')

        # Assemble colormap from given parameters
        cmap = self.config.get('cmap', None)
        # If a single color provided, prepend 'white' color, so that a resulting tuple defines binary colormap
        if is_color_like(cmap):
            cmap = ('white', cmap)
        # If a tuple of colors provided in `cmap` argument convert it into a colormap
        if isinstance(cmap, tuple):
            cmap = make_cmap(colors=cmap)
        else:
            cmap = copy(plt.get_cmap(cmap))
        # Set a color for nan/masked values display to colormap if provided
        mask_color = self.config.get('mask_color', None)
        cmap.set_bad(color=mask_color)

        image = self.ax.imshow(data, cmap=cmap, **image_config)

        return image

    def histogram(self, data):
        """ Display data as 1-D histogramogram. """
        histogram_keys = ['bins', 'color', 'alpha', 'label']
        histogram_config = filter_config(self.config, histogram_keys, prefix='histogram_')

        _, _, bar = self.ax.hist(data, **histogram_config)

        return bar

    def curve(self, data):
        """ Display data as a polygonal chain. """
        x, y = data

        curve_keys = ['color', 'linestyle', 'alpha']
        curve_config = filter_config(self.config, curve_keys, prefix='curve_')

        line = self.ax.plot(x, y, **curve_config)[0]

        return line

    def loss(self, data):
        """ Display data as a polygonal chain, optionally display running mean and learning rate with nice defaults. """
        loss, smoothed, lr = data

        label = self.config.get('label', f'loss #{self.index + 1}')
        loss_label = label + f' ⟶ {loss[-1]:2.3f}'
        final_window = self.config.get('final_window', None)
        if final_window is not None:
            final = np.mean(loss[-final_window:]) #pylint: disable=invalid-unary-operand-type
            loss_label += f"\nmean over last {final_window} iterations={final:2.3f}"

        curves = []

        curve_keys = ['color', 'linestyle', 'linewidth', 'alpha']
        loss_config = filter_config(self.config, curve_keys, prefix='curve_')
        loss_curve = self.ax.plot(loss, label=loss_label, **loss_config)
        curves.extend(loss_curve)

        if smoothed is not None:
            smoothed_color = scale_lightness(loss_config['color'], scale=.5)
            smoothed_loss_label = label + ' running mean'
            smooth_curve = self.ax.plot(smoothed, label=smoothed_loss_label, color=smoothed_color, linestyle='--')
            curves.extend(smooth_curve)

        if lr is not None:
            lr_label = f'learning rate №{self.index + 1} ⟶ {lr[-1]:.0e}'
            lr_config = filter_config(self.config, curve_keys, prefix='lr_')
            lr_curve = self.subplot.twin_ax.plot(lr, label=lr_label, **lr_config)
            self.twin_ax.set_ylabel('Learning rate', fontsize=12)
            curves.extend(lr_curve)

        return curves


class Subplot:
    """ Implements plotter subplot, managing new layers creation and axis annotation.
        Basically acts as a proxy over `matplotlib.axes`.
    """
    def __init__(self, plotter, ax):
        self.plotter = plotter
        self.ax = ax
        self._twin_ax = None
        self.layers = []
        self.config = {}
        self.annotations = {}

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.layers[key]
        raise ValueError(f"Only integer keys are supported for layers indexing, got {type(key)}.")

    @property
    def twin_ax(self):
        """ Twin axis of original subplot axis that shares the x-axis with it. Created on demand. """
        if self._twin_ax is None:
            self._twin_ax = self.ax.twinx()
        return self._twin_ax

    @property
    def empty(self):
        """ Indicator that subplot has no layers. """
        return len(self.layers) == 0

    def disable(self):
        """ Hide subplot. """
        self.ax.set_axis_off()

    def plot(self, mode, data, config):
        """ Update subplot config with given params, create subplot layers, annotate subplot. """
        self.config.update(config)

        for layer_index, layer_data in enumerate(data):
            layer_config = filter_config(config, index=layer_index)
            layer = Layer(self, mode, layer_index, layer_config, layer_data)
            self.layers.append(layer)

        annotations = self.annotate(mode)
        self.annotations.update(annotations)

    def annotate(self, mode):
        """ Apply requested annotation functions to given axis with chosen parameters. """
        # pylint: disable=too-many-branches
        annotations = {}

        text_keys = ['size', 'family', 'color']
        text_config = filter_config(self.config, text_keys, prefix='text_')

        # title
        keys = ['title', 'y']
        title_config = filter_config(self.config, keys, prefix='title_')
        label = None
        if 'label' in title_config:
            label = title_config.pop('label')
        if 'title' in title_config:
            label = title_config.pop('title')
        title_config['label'] = label
        title_config = {**text_config, **title_config}
        if title_config:
            annotations['title'] = self.ax.set_title(**title_config)

        # xlabel
        keys = ['xlabel']
        xlabel_config = filter_config(self.config, keys, prefix='xlabel_')
        xlabel_config = {**text_config, **xlabel_config}
        if xlabel_config and 'xlabel' in xlabel_config:
            annotations['xlabel'] = self.ax.set_xlabel(**xlabel_config)

        # ylabel
        keys = ['ylabel']
        ylabel_config = filter_config(self.config, keys, prefix='ylabel_')
        ylabel_config = {**text_config, **ylabel_config}
        if ylabel_config and 'ylabel' in ylabel_config:
            annotations['ylabel'] = self.ax.set_ylabel(**ylabel_config)

        # xticks
        xticks_config = filter_config(self.config, [], prefix='xticks_')
        ticks = filter_config(self.config, 'ticks')
        xticks = filter_config(self.config, 'xticks')
        xticks = ticks if ticks is not None else xticks
        if xticks is not None:
            xticks_config['ticks'] = xticks
        if xticks_config:
            self.ax.set_xticks(**xticks_config)

        # yticks
        yticks_config = filter_config(self.config, [], prefix='yticks_')
        ticks = filter_config(self.config, 'ticks')
        yticks = filter_config(self.config, 'yticks')
        yticks = ticks if ticks is not None else yticks
        if yticks is not None:
            yticks_config['ticks'] = yticks
        if yticks_config:
            self.ax.set_yticks(**yticks_config)

        # ticks
        keys = ['labeltop', 'labelright', 'labelcolor', 'direction']
        tick_config = filter_config(self.config, keys, prefix='tick_')
        if tick_config:
            self.ax.tick_params(**tick_config)

        # xlim
        xlim_config = filter_config(self.config, ['xlim'], prefix='xlim_')
        if 'xlim' in xlim_config:
            xlim_config['left'] = xlim_config.get('left', xlim_config.pop('xlim'))
        if xlim_config:
            self.ax.set_xlim(**xlim_config)

        # ylim
        ylim_config = filter_config(self.config, ['ylim'], prefix='ylim_')
        if 'ylim' in ylim_config:
            ylim_config['bottom'] = ylim_config.get('bottom', ylim_config.pop('ylim'))
        if ylim_config:
            self.ax.set_ylim(**ylim_config)

        # colorbar
        keys = ['colorbar', 'width', 'pad', 'fake', 'annotations']
        colorbar_config = filter_config(self.config, keys, prefix='colorbar_')
        if colorbar_config:
            colorbar_config['image'] = self.layers[0].object

            if 'pad' not in colorbar_config:
                pad = 0.4
                labelright = filter_config(self.config, 'labelright', prefix='tick_')
                if labelright:
                    ax_x1 = self.plotter.get_bbox(self.ax).x1
                    yticklabels = self.ax.get_yticklabels()
                    max_ytick_label_x1 = max(self.plotter.get_bbox(label).x1
                                             for label in yticklabels[len(yticklabels)//2:])
                    pad += (max_ytick_label_x1 - ax_x1) # account for width of yticklabels to the right of the subplot
                colorbar_config['pad'] = pad

            annotations['colorbar'] = self.add_colorbar(**colorbar_config)

        # legend
        if mode == 'loss':
            self.config['handles'] = [layer.object for layer in self.layers]

        keys = ['labels', 'handles', 'size', 'loc', 'ha', 'va', 'handletextpad']
        legend_config = filter_config(self.config, keys, prefix='legend_')
        if legend_config.get('labels') or legend_config.get('handles'):
            if 'cmap' in self.config:
                colors = filter_config(self.config, 'cmap')
            if 'color' in self.config:
                colors = filter_config(self.config, 'color')
            if 'colors' in legend_config:
                colors = legend_config.pop('colors')
            legend_config['colors'] = colors
            legend_config['alphas'] = self.config.get('alpha')
            annotations['legend'] = self.add_legend(mode=mode, **legend_config)

        # grid
        grid = filter_config(self.config, 'grid')
        grid_keys = ['color', 'linestyle', 'freq']

        minor_config = filter_config(self.config, grid_keys, prefix='minor_grid_')
        if grid in ('minor', 'both') and minor_config:
            self.add_grid(self.ax, grid_type='minor', **minor_config)

        major_config = filter_config(self.config, grid_keys, prefix='major_grid_')
        if grid in ('major', 'both') and minor_config:
            self.add_grid(self.ax, grid_type='major', **major_config)

        spine_colors = self.config.get('spine_color')
        if spine_colors is not None:
            spines = self.ax.spines.values()
            spine_colors = spine_colors if isinstance(spine_colors, list) else [spine_colors] * len(spines)
            for spine, color in zip(spines, spine_colors):
                spine.set_edgecolor(color)

        facecolor = self.config.get('facecolor', None)
        if facecolor is not None:
            self.ax.set_facecolor(facecolor)

        self.ax.set_axisbelow(self.config.get('set_axisbelow', False))

        if self.config.get('disable_axes'):
            self.ax.set_axis_off()
        elif not self.ax.axison:
            self.ax.set_axis_on()

        # Change scale of axis, if needed
        if self.config.get('log') or self.config.get('log_loss'):
            self.ax.set_yscale('log')

        if self.config.get('log_twin') or self.config.get('log_lr'):
            self.twin_ax.set_yscale('log')

        return annotations

    def add_colorbar(self, image, width=.2, pad=None, color='black', colorbar=False):
        """ Append colorbar to the image on the right. """
        if colorbar is False:
            return None

        divider = axes_grid1.make_axes_locatable(image.axes)
        cax = divider.append_axes("right", size=width, pad=pad)

        if colorbar is None:
            cax.set_axis_off()
            return None

        colorbar = image.axes.figure.colorbar(image, cax=cax)
        colorbar.ax.yaxis.set_tick_params(color=color, labelcolor=color)

        return colorbar

    def add_legend(self, mode='image', handles=None, labels=None, colors='none', alphas=1, size=10, **kwargs):
        """ Add patches to ax legend.

        Parameters
        ----------
        ax : int or instance of `matploblib.axes.Axes`
            Axes to put labels into. If and int, used for indexing `self.axes`.
        mode : 'image', 'histogram', 'curve', 'loss'
            Mode to match legend hadles patches to.
            If from ('image', 'histogram'), use rectangular legend patches.
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
        if (handles is None and labels is None) or (handles is not None and labels is not None):
            raise ValueError("One and only one of `handles`, `labels` must be specified.")

        # get legend that already exists
        legend = self.ax.get_legend()
        old_handles = getattr(legend, 'legendHandles', [])
        handler_map = getattr(legend, '_custom_handler_map', {})

        # make new handles
        if handles is None:
            labels = to_list(labels)
            colors = colors if isinstance(colors, list) else [colors] * len(labels)
            alphas = alphas if isinstance(alphas, list) else [alphas] * len(labels)

            new_handles = []
            for color, alpha, label in zip(colors, alphas, labels):
                if label is None:
                    continue
                if mode in ('image', 'histogram'):
                    if is_color_like(color):
                        handle = Patch(color=color, alpha=alpha, label=label)
                    else:
                        handle = PatchCollection(patches=[], cmap=color, label=label)
                        handler_map[PatchCollection] = ColorMappingHandler()
                elif mode in ('curve', 'loss'):
                    handle = Line2D(xdata=[0], ydata=[0], color=color, alpha=alpha, label=label)
                new_handles.append(handle)
        else:
            new_handles = to_list(handles)

        # extend existing handles and labels with new ones
        kwargs['handles'] = old_handles + new_handles
        legend = self.ax.legend(prop={'size': size}, handler_map=handler_map, **kwargs)

        return legend

    def add_text(self, text, size=10, x=0.5, y=0.5, ha='center', va='center', bbox='default', **kwargs):
        """ Add text to axis.

        A convenient method for adding text in box (usually on empty ax).

        Parameters
        ----------
        ax : int or instance of `matploblib.axes.Axes`
            Axes to put labels into. If an int, used for indexing `self.axes`.
        text : str
            Text to display.
        size : int
            Text size.
        x, y : float
            The position to place the text in data coordinates.
        ha : 'center', 'right', 'left'
            Text horizontal alignment.
        va : 'top', 'bottom', 'center', 'baseline', 'center_baseline'
            Text vertical alignment.
        bbox : 'default' or dict with properties for `matplotlib.patches.FancyBboxPatch`
            Properties of box containing the text.
        kwargs : misc
            For `matplotlib.legend`.
        """
        if bbox == 'default':
            bbox = {'boxstyle': 'square', 'fc': 'none'}

        return self.ax.text(x=x, y=y, s=text, size=size, ha=ha, va=va, bbox=bbox, **kwargs)

    @staticmethod
    def add_grid(ax, grid_type, x_n=None, y_n=None, zorder=0, **kwargs):
        """ Set axis grid parameters. """
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


class Plot:
    """ Multiple images plotter.

    General idea is to display graphs for provided data while passing other keyword arguments to corresponding
    `matplotlib` functions (e.g. `figsize` goes to figure initialization, `title` goes to `plt.set_title`, etc.).

    The logic behind the process is the following:
    1. Parse data:
        - Calculate subplots sizes - look through all data items
          and estimate every subplot shape taking max of all its layers shapes.
        - Put provided arrays into double nested list.
          Nestedness levels define subplot and layer data order correspondingly.
        - Infer images combination mode.
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
    mode : 'image', 'histogram', 'curve', 'loss'
        If 'image' plot given arrays as images.
        If 'histogram' plot histogramogram of flattened array.
        If 'curve' plot given arrays as curve lines.
        If 'loss' plot given arrays as loss curves.
    combine : 'overlay', 'separate' or 'mixed'
        Whether overlay images on a single axis, show them on separate ones or use mixed approach.
    kwargs :
        - For one of `ax_image`, `ax_histogram`, `ax_curve`, `ax_loss` methods (depending on chosen mode).
            Parameters and data nestedness levels must match.
            Every param with 'image_', 'histogram_', 'curve_', 'loss_' prefix is redirected to corresponding method.
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

    Parameters for 'image' mode
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
    image_{parameter} : misc
        Any parameter valid for `plt.imshow`.

    Parameters for 'histogram' mode
    ----------------------------
    color : valid matplotlib color
        Defines color to display histogramogram with.
    alpha : number in (0, 1) range
        Hisotgram opacity (0 means fully transparent, i.e. invisible, and 1 - totally opaque).
        Useful when `combine='overlay'`.
    bins : int
        Number of bins for histogramogram.
    mask_values : number or tuple of numbers
        Values that should be masked on image display.
    mask_color : valid matplotlib color
        Color to display masked values with.
    histogram_{parameter} : misc
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
    Address docs of `ax_image``, `ax_histogram`, `ax_curve`, `ax_loss` and `annotate_axis` for details.

    This also allows one to pass arguments of the same name for different plotting steps.
    E.g. `plt.set_title` and `plt.set_xlabel` both require `size` argument.
    Providing `{'size': 30}` in kwargs will affect both title and x-axis labels.
    To change parameter for title only, one can provide {'title_fontsize': 30}` instead.
    """
    def __init__(self, data=None, combine='overlay', mode='image', **kwargs):
        self.figure = None
        self.subplots = None

        self.plot(data=data, combine=combine, mode=mode, **kwargs)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.subplots[key]
        raise ValueError(f"Only integer keys are supported for subplots indexing, got {type(key)}.")

    @staticmethod
    def parse_tuple(data, mode):
        """ Validate that tuple data item is provided with correct plot mode and convert its objects to arrays. """
        if mode not in ('curve', 'loss'):
            msg = "Tuple is a valid data item only in modes ('curve', 'loss')."
            raise ValueError(msg)
        return tuple(np.array(item) for item in data)

    @staticmethod
    def parse_array(data, mode):
        """ Validate that data dimensionality is correct for given plot mode. """
        if data.ndim == 1:
            if mode == 'image':
                return data.reshape(-1, 1)
            if mode == 'curve':
                return (range(len(data)), data)
            if mode == 'loss':
                return (data, None)

        if data.ndim == 3:
            if mode in ('curve', 'loss'):
                msg = f"In `mode={mode}` array must be 1- or 2-dimensional, got array with ndim={data.ndim}."
                raise ValueError(msg)

        if data.ndim > 3:
            if mode != 'histogram':
                msg = f"In `mode={mode}` array must be 1-, 2- or 3-dimensional, got array with ndim={data.ndim}."
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
        n_subplots = 0

        data_list = []
        if data is None:
            data_list = []
            n_subplots = 1
        elif isinstance(data, tuple):
            data_list = [[cls.parse_tuple(data=data, mode=mode)]]
            n_subplots = 1
        elif isinstance(data, np.ndarray):
            data_list = [[cls.parse_array(data=data, mode=mode)]]
            n_subplots = 1
        elif isinstance(data, list) and contains_numbers(data):
            data_list = [[np.array(data)]]
            n_subplots = 1
        elif isinstance(data, list):
            if any(isinstance(item, list) and not contains_numbers(item) for item in data):
                combine = 'mixed'

            data_list = []
            for item in data:

                if item is None:
                    if combine == 'overlay':
                        msg = "`None` is a placeholder future subplots. It makes not sense when `combine='overlay'`."
                        raise ValueError(msg)
                    data_item = None
                elif isinstance(item, tuple):
                    data_item = [cls.parse_tuple(data=item, mode=mode)]
                elif isinstance(item, np.ndarray):
                    data_item = [cls.parse_array(data=item, mode=mode)]
                elif isinstance(item, list) and contains_numbers(item):
                    data_item = [np.array(item)]
                elif isinstance(item, list):
                    if combine == 'separate':
                        raise ValueError("Data list items cant be lists themselves when `combine='separate'`")
                    data_item = []
                    for subitem in item:
                        if isinstance(subitem, tuple):
                            data_item += [cls.parse_tuple(data=subitem, mode=mode)]
                        elif isinstance(subitem, np.ndarray):
                            data_item += [cls.parse_array(data=subitem, mode=mode)]
                        elif isinstance(subitem, list) and contains_numbers(subitem):
                            data_item += [np.array(subitem)]
                        elif isinstance(subitem, list):
                            raise ValueError("!!.")
                else:
                    msg = f"Valid data items are None, tuple, array or list of those, got {type(item)}."
                    raise ValueError(msg)

                if combine in ('overlay',):
                    data_list.extend(data_item)
                elif combine in ('separate', 'mixed'):
                    data_list.append(data_item)
                    n_subplots += 1
                else:
                    msg = f"Valid combine modes are 'overlay', 'separate', 'mixed', got {combine}."
                    raise ValueError(msg)

            if combine == 'overlay':
                data_list = [data_list]
                n_subplots = 1

        return data_list, combine, n_subplots

    def make_default_config(self, mode, n_subplots, data, ncols=None, ratio=None, scale=1, max_fig_width=25,
                            nrows=None, xlim=(None, None), ylim=(None, None), **kwargs):
        """ Infer default figure params from shapes of provided data. """
        config = {'tight_layout': True, 'facecolor': 'snow'}

        if mode in ('image', 'histogram'):
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

        if mode in ('image', 'histogram'):
            fig_width = 8 * ncols * scale
        elif mode in ('curve', 'loss'):
            fig_width = 16 * ncols * scale

        # Make figsize
        if ratio is None:
            if mode == 'image':
                if not isinstance(xlim, list):
                    xlim = [xlim] * n_subplots
                if not isinstance(ylim, list):
                    ylim = [ylim] * n_subplots

                widths = []
                heights = []

                shapes = [subplot_data[0].shape if subplot_data is not None else None for subplot_data in data]
                for idx, shape in enumerate(shapes):
                    if shape is None:
                        continue

                    order_axes = filter_config(kwargs, 'order_axes', index=idx)
                    order_axes = order_axes or self.IMAGE_DEFAULTS['order_axes']

                    min_height = ylim[idx][0] or 0
                    max_height = ylim[idx][1] or shape[order_axes[0]]
                    subplot_height = abs(max_height - min_height)
                    heights.append(subplot_height)

                    min_width = xlim[idx][0] or shape[order_axes[1]]
                    max_width = xlim[idx][1] or 0
                    subplot_width = abs(max_width - min_width)
                    widths.append(subplot_width)

                mean_height, mean_width = np.mean(heights), np.mean(widths)
                if np.isnan(mean_height) or np.isnan(mean_width):
                    ratio = 1
                else:
                    ratio = (mean_height * 1.05 * nrows) / (mean_width * 1.05 * ncols)

            elif mode == 'histogram':
                ratio = 2 / 3 / ncols * nrows

            elif mode in ('curve', 'loss'):
                ratio = 1 / 3 / ncols * nrows

        fig_height = fig_width * ratio

        if fig_width > max_fig_width:
            fig_width = max_fig_width
            fig_height = fig_width * ratio

        config['figsize'] = (fig_width, fig_height)

        return config

    def make_figure(self, mode, n_subplots, data, axes=None, figure=None, **kwargs):
        """ Create figure and axes if needed. """
        if axes is None and figure is not None:
            axes = figure.axes

        if axes is None:
            default_config = self.make_default_config(mode=mode, n_subplots=n_subplots, data=data, **kwargs)
            subplots_keys = ['figsize', 'facecolor', 'dpi', 'ncols', 'nrows', 'tight_layout', 'gridspec_kw']
            config = filter_config(kwargs, subplots_keys, prefix='figure_')
            config = {**default_config, **config}

            figure, axes = plt.subplots(**config)
            axes = to_list(axes)
        else:
            axes = to_list(axes)
            figure = axes[0].figure
            config = {}

            if len(axes) < n_subplots:
                raise ValueError(f"Not enough axes provided — got ({len(axes)}) for {n_subplots} subplots.")

        return figure, axes, config

    def get_bbox(self, obj):
        """ Get object bounding box in inches. """
        renderer = self.figure.canvas.get_renderer()
        transformer = self.figure.dpi_scale_trans.inverted()
        return obj.get_window_extent(renderer=renderer).transformed(transformer)

    def adjust_figsize(self):
        """ Look through subplots annotation objects and add figsize corrections for their widths and heights. """
        ncols, nrows = self.figure.axes[0].get_subplotspec().get_gridspec().get_geometry()
        fig_width, fig_height = self.figure.get_size_inches()

        extra_width = 0
        extra_height = 0
        if 'suptitle' in self.figure_objects:
            suptitle_obj = self.figure_objects['suptitle']
            suptitle_height = self.get_bbox(suptitle_obj).height
            extra_height += suptitle_height

        ax_widths = []
        ax_heights = []
        for subplot in self.subplots:
            ax = subplot.ax
            annotations = subplot.annotations

            width = 0
            height = 0
            if annotations is not None:
                ax_bbox = self.get_bbox(ax)

                if 'title' in annotations:
                    title_obj = annotations['title']
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

                if 'xlabel' in annotations:
                    xlabel_obj = annotations['xlabel']
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

                if 'ylabel' in annotations:
                    ylabel_obj = annotations['ylabel']
                    ylabel_width = self.get_bbox(ylabel_obj).width
                    width += ylabel_width

            ax_widths.append(width)
            ax_heights.append(height)

        ax_widths = np.array(ax_widths).reshape(nrows, ncols)
        extra_width += ax_widths.max(axis=1).sum()

        ax_heights = np.array(ax_heights).reshape(nrows, ncols)
        extra_height += ax_heights.max(axis=0).sum()

        new_figsize = (fig_width + extra_width, fig_height + extra_height)
        self.figure.set_size_inches(new_figsize)

    MASK_COLORS = ['firebrick', 'mediumseagreen', 'thistle', 'darkorange', 'navy', 'gold',
                    'red', 'turquoise', 'darkorchid', 'darkkhaki', 'royalblue', 'yellow',
                    'chocolate', 'forestgreen', 'lightpink', 'darkslategray', 'deepskyblue', 'wheat']

    IMAGE_DEFAULTS = {
        # image
        'cmap': CycledList(['Greys_r'] + MASK_COLORS, cycle_from=1),
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

    HISTOGRAM_DEFAULTS = {
        # preprocessing
        'flatten': True,
        # histogram
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

    CURVE_COLORS = ['cornflowerblue', 'sandybrown', 'lightpink', 'mediumseagreen', 'thistogramle', 'firebrick',
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

    def plot(self, data=None, combine='overlay', mode='image', save=False, show=True,
             adjust_figsize='image', axes=None, axis=None, ax=None, subplots=None, **kwargs):
        """ Plot data on subplots.

        Parses axes from kwargs if provided, else creates them.
        Filters parameters and calls chosen plot method for every axis-data pair.
        """
        self.config = {**self.ANNOTATION_DEFAULTS, **kwargs}

        data, combine, n_subplots = self.parse_data(data=data, combine=combine, mode=mode)

        axes = axes or axis or ax
        if self.subplots is None:
            fig, axes, fig_config = self.make_figure(mode=mode, n_subplots=n_subplots, data=data, axes=axes, **kwargs)
            self.figure, self.figure_config = fig, fig_config
            self.subplots = [Subplot(self, ax) for ax in axes]
            active_subplots = subplots or self.subplots
        else:
            if axes is not None:
                msg = "Subplots already created and new axes cannot be specified."
                raise ValueError(msg)
            active_subplots = [self.subplots[idx] for idx in to_list(subplots)]

        mode_defaults = getattr(self, f"{mode.upper()}_DEFAULTS")

        for idx, subplot in enumerate(active_subplots):
            subplot_data = data[idx] if idx < len(data) else None

            if subplot_data is None:
                if subplot.empty:
                    subplot.disable()
                continue

            subplot_idx = None if combine == 'overlay' else idx
            subplot_config = filter_config(self.config, index=subplot_idx)
            subplot_config = {**mode_defaults, **subplot_config}

            subplot.plot(mode, subplot_data, subplot_config)

        figure_objects = self.annotate()
        self.figure_objects = figure_objects

        if adjust_figsize == mode or adjust_figsize is True:
            self.adjust_figsize()

        if show:
            self.show()
        else:
            self.close()

        if save or 'savepath' in kwargs:
            self.save(**kwargs)

        return self

    def __call__(self, mode, **kwargs):
        self.plot(mode=mode, **kwargs)

    def __repr__(self):
        return ''

    def __str__(self):
        return f"<Batchflow Plotter with {len(self.axes)} axes>"

    def annotate(self):
        """ Put suptitle with given parameters over figure and apply `tight_layout`. """
        annotations = {}

        text_keys = ['size', 'family', 'color']
        text_config = filter_config(self.config, text_keys, prefix='text_')

        # suptitle
        keys = ['suptitle', 't', 'y']
        suptitle_config = filter_config(self.config, keys, prefix='suptitle_')
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
            annotations['suptitle'] = annotations['suptitle'] = self.figure.suptitle(**suptitle_config)

        self.figure.tight_layout()

        return annotations

    def show(self):
        self.figure = plt.figure(self.figure)

    def save(self, **kwargs):
        """ Save plot. """
        default_config = {
            'savepath': datetime.now().strftime('%Y-%m-%d_%H:%M:%S.png'),
            'bbox_inches': 'tight',
            'pad_inches': 0,
            'dpi': 100
        }

        save_keys = ['savepath', 'bbox_inches', 'pad_inches', 'dpi']
        save_config = filter_config(kwargs, save_keys, prefix='save_')
        save_config = {**default_config, **save_config}
        savepath = save_config.pop('savepath')

        self.figure.savefig(fname=savepath, **save_config)

    def close(self):
        """ Close figure. """
        plt.close(self.figure)


def plot_image(data, **kwargs):
    """ Shorthand for image plotting. """
    return Plot(data, mode='image', **kwargs)

def plot_histogram(data, **kwargs):
    """ Shorthand for histogramogram plotting. """
    return Plot(data, mode='histogram', **kwargs)

def plot_curve(data, **kwargs):
    """ Shorthand for curve plotting. """
    return Plot(data, mode='curve', **kwargs)

def plot_loss(data, **kwargs):
    """ Shorthand for loss plotting. """
    return Plot(data, mode='loss', **kwargs)
