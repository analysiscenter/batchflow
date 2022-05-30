""" Plot functions. """
from ast import literal_eval
from colorsys import rgb_to_hls, hls_to_rgb
from copy import copy
from datetime import datetime
from numbers import Number
import operator


import cv2
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
    """ Find comparison operator in string and apply it against given argument and those parsed from the string
    to the right of the found operator.

    Used for evaluation of string expressions that are provided as masking conditions for visualized data.

    Examples
    --------
    >>> evaluate_str_comparison(np.arange(5), '<3')
    array([ True,  True,  True, False, False])
    """
    for key in STR_TO_OPERATION:
        if key in string:
            operation = STR_TO_OPERATION[key]
            arg1 = literal_eval(string.split(key)[-1])
            return operation(arg0, arg1)
    msg = f"Given string '{string}' does not contain any of supported operators: {list(STR_TO_OPERATION.keys())}"
    raise ValueError(msg)


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

class PlotConfig(dict):
    """ Dictionary with additional indexing and filtering capabilities. """
    def maybe_index(self, key, index):
        """ Get i-th element of parameter if index is provided and parameter value is a list else return it unchanged.

        Parameters
        ----------
        key : str
            Parameter name.
        value : misc
            Parameter value.
        index : None or int
            If not None, a number to use for parameter indexing.
        index : int
            Index of argument value to retrieve.
            If none provided, get whole argument value.
            If value is non-indexable, get it without indexing.

        Raises
        ------
        ValueError
            If parameter is a list but the index is greater than its length.
        """
        value = self[key]

        if index is not None and isinstance(value, list):
            try:
                return value[index]
            except IndexError as e:
                msg = f"Tried to obtain element #{index} from `{key}={value}`. Either provide parameter value "\
                        f"as a single item (to use the same `{key}` several times) or add more elements to it."
                raise ValueError(msg) from e
        return value

    def __getitem__(self, index):
        if index is None:
            return self
        if isinstance(index, int):
            return type(self)({key: self.maybe_index(key, index) for key in self})
        return super().__getitem__(index)

    def filter(self, keys=None, prefix=''):
        """ Make a subconfig of parameters with required keys.

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
        """
        if keys is None:
            keys = list(self.keys())
        elif prefix:
            keys += [key.split(prefix)[1] for key in self if key.startswith(prefix)]

        result = type(self)()

        for key in keys:
            if prefix + key in self:
                result[key] = self[prefix + key]
            elif key in self:
                result[key] = self[key]

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

        if mode in ('image', 'histogram') and 'mask' not in self.config:
            # Add `0` to a list of values that shouldn't be displayed if image is a binary mask
            if is_binary(data.flatten()):
                self.config['mask'] = 0
                self.config['vmin'] = 0
                self.config['vmax'] = 1

        preprocessed_data = self.preprocess(data)
        self.object = getattr(self, mode)(preprocessed_data)

    def transpose(self, data):
        """ Change array axes order if needed. """
        transpose = self.config.get('transpose', None)
        if transpose is not None:
            transpose = transpose[:data.ndim]
            data = np.transpose(data, transpose)
        return data

    def flatten(self, data):
        """ Make array 1d if needed. """
        flatten = self.config.get('flatten', False)
        if flatten:
            data = data.flatten()
        return data

    def dilate(self, data):
        """ Apply dilation to array. """
        dilation_config = self.config.get('dilate', False)
        if dilation_config:
            if dilation_config is True:
                dilation_config = {'kernel': np.ones((3, 1), dtype=np.uint8)}
            elif isinstance(dilation_config, int):
                dilation_config = {'iterations': dilation_config}
            elif isinstance(dilation_config, tuple):
                dilation_config = {'kernel': np.ones(dilation_config, dtype=np.uint8)}

            data = cv2.dilate(data, **dilation_config)
        return data

    def mask(self, data):
        """ Mask array values matching given conditions. """
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
        return data

    def preprocess(self, data):
        """ Look through layer config for requested data transformations and apply them. """
        if self.mode == 'image':
            data = self.transpose(data)
            data = self.dilate(data)
            data = self.mask(data)

        if self.mode == 'histogram':
            data = self.flatten(data)
            data = self.mask(data)

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
            if window is None or loss is None:
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

        image_keys = ['alpha', 'vmin', 'vmax', 'extent']
        image_config = self.config.filter(image_keys, prefix='image_')

        image = self.ax.imshow(data, cmap=cmap, **image_config)

        return image

    def histogram(self, data):
        """ Display data as 1-D histogram. """
        histogram_keys = ['bins', 'color', 'alpha', 'label']
        histogram_config = self.config.filter(histogram_keys, prefix='histogram_')

        _, _, bar = self.ax.hist(data, **histogram_config)

        return bar

    def curve(self, data):
        """ Display data as a polygonal chain. """
        x, y = data

        curve_keys = ['color', 'linestyle', 'alpha', 'label']
        curve_config = self.config.filter(curve_keys, prefix='curve_')

        line = self.ax.plot(x, y, **curve_config)[0]

        return line

    def loss(self, data):
        """ Display a combination of loss curve, its smoothed version and learning rate with nice defaults. """
        loss, smoothed, lr = data

        curves = []

        curve_keys = ['color', 'linestyle', 'linewidth', 'alpha']

        if loss is not None:
            label = self.config.get('label', f'loss #{self.index + 1}')
            loss_label = label + f' ⟶ {loss[-1]:2.3f}'
            final_window = self.config.get('final_window', None)
            if final_window is not None:
                final = np.mean(loss[-final_window:]) #pylint: disable=invalid-unary-operand-type
                loss_label += f"\nmean over last {final_window} iterations={final:2.3f}"

            loss_config = self.config.filter(curve_keys, prefix='curve_')
            loss_curve = self.ax.plot(loss, label=loss_label, **loss_config)
            curves.extend(loss_curve)

        if smoothed is not None:
            smoothed_color = scale_lightness(loss_config['color'], scale=.5)
            smoothed_loss_label = label + ' running mean'
            smooth_curve = self.ax.plot(smoothed, label=smoothed_loss_label, color=smoothed_color, linestyle='--')
            curves.extend(smooth_curve)

        if lr is not None:
            lr_label = f'learning rate №{self.index + 1} ⟶ {lr[-1]:.0e}'
            lr_config = self.config.filter(curve_keys, prefix='lr_')
            lr_ax = self.ax if loss is None else self.twin_ax
            lr_curve = lr_ax.plot(lr, label=lr_label, **lr_config)
            lr_ax.set_ylabel('Learning rate', fontsize=12)
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
        self.config = None
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
        """ Update subplot config with given parameters, for every data item create and delegate data plotting to it
        with parameters relevant to this subplot, annotate subplot (add title, labels, colorbar, grid etc.).
        """
        self.config = config

        for layer_index, layer_data in enumerate(data):
            layer_config = config[layer_index]
            layer = Layer(self, mode, layer_index, layer_config, layer_data)
            self.layers.append(layer)

        annotations = self.annotate(mode)
        self.annotations.update(annotations)

    def annotate(self, mode):
        """ Apply requested annotation functions to subplot axis with parameters from subplot config. """
        # pylint: disable=too-many-branches
        annotations = {}

        text_keys = ['size', 'family', 'color']
        text_config = self.config.filter(text_keys, prefix='text_')

        # title
        keys = ['title', 'y']
        title_config = self.config.filter(keys, prefix='title_')
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
        xlabel_config = self.config.filter(keys, prefix='xlabel_')
        xlabel_config = {**text_config, **xlabel_config}
        if xlabel_config and 'xlabel' in xlabel_config:
            annotations['xlabel'] = self.ax.set_xlabel(**xlabel_config)

        # ylabel
        keys = ['ylabel']
        ylabel_config = self.config.filter(keys, prefix='ylabel_')
        ylabel_config = {**text_config, **ylabel_config}
        if ylabel_config and 'ylabel' in ylabel_config:
            annotations['ylabel'] = self.ax.set_ylabel(**ylabel_config)

        # xticks
        xticks_config = self.config.filter([], prefix='xticks_')
        ticks = self.config.get('ticks', None)
        xticks = self.config.get('xticks', None)
        xticks = ticks if ticks is not None else xticks
        if xticks is not None:
            xticks_config['ticks'] = xticks
        if xticks_config:
            self.ax.set_xticks(**xticks_config)

        # yticks
        yticks_config = self.config.filter([], prefix='yticks_')
        ticks = self.config.get('ticks', None)
        yticks = self.config.get('yticks', None)
        yticks = ticks if ticks is not None else yticks
        if yticks is not None:
            yticks_config['ticks'] = yticks
        if yticks_config:
            self.ax.set_yticks(**yticks_config)

        # ticks
        keys = ['labeltop', 'labelright', 'labelcolor', 'direction']
        tick_config = self.config.filter(keys, prefix='tick_')
        if tick_config:
            self.ax.tick_params(**tick_config)

        # xlim
        xlim_config = self.config.filter(['xlim'], prefix='xlim_')
        if 'xlim' in xlim_config:
            xlim_config['left'] = xlim_config.get('left', xlim_config.pop('xlim'))
        if xlim_config:
            self.ax.set_xlim(**xlim_config)

        # ylim
        ylim_config = self.config.filter(['ylim'], prefix='ylim_')
        if 'ylim' in ylim_config:
            ylim_config['bottom'] = ylim_config.get('bottom', ylim_config.pop('ylim'))
        if ylim_config:
            self.ax.set_ylim(**ylim_config)

        # colorbar
        keys = ['colorbar', 'width', 'pad', 'fake', 'annotations']
        colorbar_config = self.config.filter(keys, prefix='colorbar_')
        if colorbar_config:
            colorbar_config['image'] = self.layers[0].object

            if 'pad' not in colorbar_config:
                pad = 0.4
                labelright = self.config.get('labelright', None)
                if labelright:
                    ax_x1 = self.plotter.get_bbox(self.ax).x1
                    yticklabels = self.ax.get_yticklabels()
                    max_ytick_label_x1 = max(self.plotter.get_bbox(label).x1
                                             for label in yticklabels[len(yticklabels)//2:])
                    pad += (max_ytick_label_x1 - ax_x1) # account for width of yticklabels to the right of the subplot
                colorbar_config['pad'] = pad

            annotations['colorbar'] = self.add_colorbar(**colorbar_config)

        # legend
        if mode in ('loss', 'curve'):
            self.config['label'] = [layer.object for layer in self.layers]

        label = self.config.get('label')
        if label is not None:
            color = self.config.get('cmap') or self.config.get('color')
            alpha = self.config.get('alpha', 1)

            legend_keys = ['size', 'loc', 'ha', 'va', 'handletextpad']
            legend_config = self.config.filter(legend_keys, prefix='legend_')

            annotations['legend'] = self.add_legend(mode=mode, label=label, color=color, alpha=alpha, **legend_config)

        # grid
        grid = self.config.get('grid', None)
        grid_keys = ['color', 'linestyle', 'freq']

        minor_config = self.config.filter(grid_keys, prefix='minor_grid_')
        if grid in ('minor', 'both') and minor_config:
            self.add_grid(self.ax, grid_type='minor', **minor_config)

        major_config = self.config.filter(grid_keys, prefix='major_grid_')
        if grid in ('major', 'both') and minor_config:
            self.add_grid(self.ax, grid_type='major', **major_config)

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

    def add_legend(self, mode='image', label=None, color='none', alpha=1, size=10, **kwargs):
        """ Add patches to ax legend.

        Parameters
        ----------
        ax : int or instance of `matploblib.axes.Axes`
            Axes to put labels into. If and int, used for indexing `self.axes`.
        mode : 'image', 'histogram', 'curve', 'loss'
            Mode to match legend hadles patches to.
            If from ('image', 'histogram'), use rectangular legend patches.
            If from ('curve', 'loss'), use line legend patches.
        labels : str, Artist or list
            If str, a text to show next to the legend patch/line.
            If Artist, must be valid handle for `plt.legend` (line, patch etc.)
            If a list, can contain objects of types described above.
        color : str, matplotlib colormap object or tuple
            Color to use for patches creation if those are not provided explicitly.
            If str. Must be valid matplotlib color (e.g. 'salmon', '#120FA3', (0.3, 0.4, 0.5)).
        alpha : number or list of numbers from 0 to 1
            Legend handles opacity.
        size : int
            Legend size.
        kwargs : misc
            For `matplotlib.legend`.
        """
        # get legend that already exists
        legend = self.ax.get_legend()
        old_handles = getattr(legend, 'legendHandles', [])
        handler_map = getattr(legend, '_custom_handler_map', {})

        # make new handles
        new_handles = []
        labels = to_list(label)
        colors = color if isinstance(color, list) else [color] * len(labels)
        alphas = alpha if isinstance(alpha, list) else [alpha] * len(labels)

        for label_item, label_color, label_alpha in zip(labels, colors, alphas):
            if label is None:
                continue
            if isinstance(label, str):
                if mode in ('image', 'histogram'):
                    if is_color_like(label_color):
                        handle = Patch(color=label_color, alpha=label_alpha, label=label_item)
                    else:
                        handle = PatchCollection(patches=[], cmap=label_color, label=label_item)
                        handler_map[PatchCollection] = ColorMappingHandler()
                elif mode in ('curve', 'loss'):
                    handle = Line2D(xdata=[0], ydata=[0], color=label_color, alpha=label_alpha, label=label_item)
                new_handles.append(handle)
            elif not label.get_label().startswith('_'):
                new_handles.append(label_item)

        if len(new_handles) > 0:
            # extend existing handles and labels with new ones
            handles = old_handles + new_handles
            legend = self.ax.legend(prop={'size': size}, handles=handles,  handler_map=handler_map, **kwargs)

        return legend

    def add_text(self, text, size=10, x=0.5, y=0.5, ha='center', va='center', bbox='default', **kwargs):
        """ Add text to subplot.

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
        """ Set parameters for subplot axis grid. """
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

    General idea is to display graphs and annotate them based on config of provided parameters for various
    `matplotlib` functions (e.g. `figsize` goes to figure initialization, `title` goes to `plt.set_title`, etc.).

    The logic behind the process is the following:
    1. Parse data — put provided arrays into double nested list. Nestedness levels define subplot and layer data order
       correspondingly. Also infer images combination mode — either overlay, separate or mixed.
    2. Parse figure axes if provided, else create them.
    3. Obtain default config for chosen plotting mode and update it with provided parameters.
    4. For every data item choose corresponding subplot and delegate data plotting to it.
    5. Annotate figure.
    6. Save plot.

    General parameters
    ----------
    data : np.ndarray or a list of np.ndarray objects or a list of lists of np.ndarray
        If list if flat, 'overlay/separate' logic is handled via `combine` parameter.
        If list is nested, outer level defines subplots order while inner one defines layers order.
        Shape of data items must match chosen plotting mode (see below).
    mode : 'image', 'histogram', 'curve', 'loss'
        If 'image' plot given arrays as images.
        If 'histogram' plot histogram of flattened array.
        If 'curve' plot given arrays as curve lines.
        If 'loss' plot given arrays as loss curves.
    combine : 'overlay', 'separate' or 'mixed'
        Whether overlay images on a single subplot, show them on separate ones or use mixed approach.
    kwargs :
        - For one of `image`, `histogram`, `curve`, `loss` methods (depending on chosen mode).
            Parameters and data nestedness levels must match.
            Every param with 'image_', 'histogram_', 'curve_', 'loss_' prefix is redirected to corresponding method.
            See detailed parameters listings below.
        - For `annotate`.
            Every param with 'title_', 'suptitle_', 'xlabel_', 'ylabel_', 'xticks_', 'yticks_', 'xlim_', 'ylim_',
            colorbar_', 'legend_' or 'grid_' prefix is redirected to corresponding matplotlib method.
            Also 'facecolor', 'set_axisbelow', 'disable_axes' arguments are accepted.

    Parameters for figure creation
    ------------------------------
    figsize : tuple
        Size of displayed figure. If not provided, infered from data shapes.
    facecolor : string or tuple of 3 or 4 numbers
        Figure background color. Must be valid matplotlib color (e.g. 'salmon', '#120FA3', (0.3, 0.4, 0.5)).
    dpi : float
        The resolution of the figure in dots-per-inch.
    ncols, nrows : int
        Number of figure columns/rows.
    tight_layout : bool
        If True adjust subplot parameters using tight_layout with default padding.
    figure_{parameter} : misc
        Any parameter valid for `plt.subplots`

    Parameters for 'image' mode
    ---------------------------
    transpose: tuple
        Order of axes for displayed images.
    dilate : bool, int, tuple of two ints or dict
        Parameter for image dilation via `cv2.dilate`.
        If bool, indicates whether image should be dilated once with default kernel (`np.ones((1,3))`).
        If int, indcates how many times image should be dilate with default kernel.
        If tuple of two ints, defines shape of kernel image should be dilate with.
        If dict, must contain keyword arguments for `cv2.dilate`.
    mask : number, str, callable or tuple of any of them
        Parameter indicating which values should be masked.
        If a number, mask this value in data.
        If str, must consists of operator and a number (e.g. '<0.5', '==2', '>=1000').
        If a callable, must return boolean mask with the same shape as original image that mark image pixels to mask.
        If a tuple, contain any combination of items of types above.
    cmap : str or matplotlib colormap object
        Сolormap to display single-channel images with. Must be valid matplotlib colormap (e.g. 'ocean', 'tab20b').
    mask_color : string or tuple of 3 or 4 numbers
        Color to display masked values with. Must be valid matplotlib color (e.g. 'salmon', '#120FA3', (0.3, 0.4, 0.5)).
    alpha : number in (0, 1) range
        Image opacity (0 means fully transparent, i.e. invisible, 1 - totally opaque). Useful when `combine='overlay'`.
    vmin, vmax : number
        Limits for normalizing image into (0, 1) range. Values beyond range are clipped (default matplotlib behaviour).
    extent : tuple of 4 numbers
        The bounding box in data coordinates that the image will fill.
    image_{parameter} : misc
        Any parameter valid for `plt.imshow`.

    Parameters for 'histogram' mode
    -------------------------------
    flatten : bool
        Whether convert input array to 1d before plot. Default is True.
    mask : number, str, callable or tuple of any of them
        Parameter indicating which values should be masked.
        If a number, mask this value in data.
        If str, must consists of operator and a number (e.g. '<0.5', '==2', '>=1000').
        If a callable, must return boolean mask with the same shape as original image that mark image pixels to mask.
        If a tuple, contain any combination of items of types above.
    color : string or tuple of 3 or 4 numbers
        Color to display histogram with. Must be valid matplotlib color (e.g. 'salmon', '#120FA3', (0.3, 0.4, 0.5)).
    alpha : number in (0, 1) range
        Histogram opacity (0 means fully transparent, i.e. invisible, and 1 - totally opaque).
        Useful when `combine='overlay'`.
    bins : int
        Number of bins for histogram.
    histogram_{parameter} : misc
        Any parameter valid for `plt.hist`.

    Parameters for 'curve' mode
    ---------------------------
    color : string or tuple of 3 or 4 numbers
        Color to display curve with. Must be valid matplotlib color (e.g. 'salmon', '#120FA3', (0.3, 0.4, 0.5)).
    linestyle : str
        Style to display curve with. Must be valid matplotlib line style (e.g. 'dashed', ':').
    alpha : number in (0, 1) range
        Curve opacity (0 means fully transparent, i.e. invisible, 1 - totally opaque). Useful when `combine='overlay'`.
    curve_{parameter} : misc
        Any parameter valid for `plt.plot`.

    Parameters for 'loss' mode
    ----------------------------
    color : string or tuple of 3 or 4 numbers
        Color to display loss curve with. Must be valid matplotlib color (e.g. 'salmon', '#120FA3', (0.3, 0.4, 0.5)).
    linestyle : str
        Style to display loss curve with. Must be valid matplotlib line style (e.g. 'dashed', ':').
    linewidth : number
        Width of loss curve.
    alpha : number in (0, 1) range
        Curve opacity (0 means fully transparent, i.e. invisible, 1 - totally opaque). Useful when `combine='overlay'`.
    window : None or int
        Size of the window to use for moving average calculation of loss curve.
    loss_{parameter}, lr_{parameter} : misc
        Any parameter valid for `plt.plot`.

    Parameters for axes annotation
    ------------------------------
    {text_object}_label: str
        Value of axes text object. Valid objects are 'suptitle', 'title', 'xlabel', 'ylabel', 'legend'.
    {text_object}_color : str or tuple
        Color of axes text object.
        Valid objects are 'suptitle', 'title', 'xlabel', 'ylabel', 'legend'.
        If str. Must be valid matplotlib colormap.
        If tuple. Must be a valid rgb color.
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
    1. The simplest one if when one provide a single data array — in that case data is displayed on a single subplot:
       >>> Plot(array)
    2. A more advanced use case is when one provide a list of arrays — plot behaviour depends on `combine` parameter:
       a. Images are put on same subplot and overlaid one over another if `combine='overlay'` (which is default):
          >>> Plot([image_0, mask_0])
       b. Images are put on separate subplots if `combine='separate'`.
          >>> Plot([image_0, image_1], combine='separate')
    3. The most complex scenario is displaying images in a 'mixed' manner (ones overlaid and others separated).
       For example, to overlay first two images but to display the third one separately, use the following notation:
       >>> Plot([[image_0, mask_0], image_1]); (`combine='mixed'` is set automatically if data is double-nested).

    The order of arrays inside the double-nested structure basically declares, which of them belong to the same subplot
    and therefore should be rendered one over another, and which must be displayed separately.

    If a parameter is provided in a list, each subplot uses its item on position corresponding to its index and
    every subplot layer in turn uses item from that sublist on positions that correspond to its index w.r.t. to subplot.
    Therefore, such parameters must resemble data nestedness level, since that allows binding subplots and parameters.
    However, it's possible for parameter to be a single item — in that case it's shared across all subplots and layers.

    For example, to display two images separately with same colormap, the following code required:
    >>> Plot([image_0, image_1], cmap='viridis')
    If one wish to use different colormaps for every image, the code should be like this:
    >>> Plot([image_0, image_1], cmap=['viridis', 'magma'])
    Finally, if a more complex data provided, the parameter nestedness level must resemble the one in data:
    >>> Plot([[image_0, mask_0], [image_1, mask_1]], cmap=[['viridis', 'red'], ['magma', 'green']])

    Advanced parameters managing
    ----------------------------
    The list of parameters expected by specific plot method is limited to the ones most frequently used.
    However there is a way to provide any parameter to a plot method, using specific prefix:
    - 'image_' — for `Axes.imshow`
    — 'histogram_' — for `Axes.hist`
    - 'curve_' —  for `Axes.plot`
    - 'loss_' — for `Axes.plot`
    - 'curve_', 'lr_' — for `Axes.plot`
    - 'text_' — for every text object (title, label etc.)
    - 'title_' — for `Axes.set_title`
    - 'xlabel_' — for `Axes.set_xlabel`
    - 'ylabel_' — for `Axes.set_ylabel`
    - 'xticks_' — for `Axes.set_xticks`
    - 'yticks_' — for `Axes.set_yticks`
    - 'tick_' — for `Axes.tick_params`
    - 'xlim_' — for `Axes.set_xlim`
    - 'ylim_' — for `Axes.set_ylim`
    - 'colorbar_' — for `Axes.colorbar`
    - 'legend_' — for `Axes.legend`
    - 'minor_grid_', 'major_grid_' — `Axes.grid`

    This also allows one to pass arguments of the same name for different plotting steps.
    E.g. `plt.set_title` and `plt.set_xlabel` both require `size` argument.
    Providing `{'size': 30}` in kwargs will affect both title and x-axis labels.
    To change parameter for title only, one can provide {'title_fontsize': 30}` instead.
    """
    def __init__(self, data=None, combine='overlay', mode='image', **kwargs):
        self.figure = None
        self.subplots = None
        self.config = {}

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
        return tuple(None if item is None else np.array(item) for item in data)

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
        """ Infer default figure parameters from shapes of provided data. """
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

                    transpose = PlotConfig(kwargs)[idx].get('transpose', None)
                    transpose = transpose or self.IMAGE_DEFAULTS['transpose']

                    min_height = ylim[idx][0] or 0
                    max_height = ylim[idx][1] or shape[transpose[0]]
                    subplot_height = abs(max_height - min_height)
                    heights.append(subplot_height)

                    min_width = xlim[idx][0] or shape[transpose[1]]
                    max_width = xlim[idx][1] or 0
                    subplot_width = abs(max_width - min_width)
                    widths.append(subplot_width)

                mean_height, mean_width = np.nanmean(heights), np.nanmean(widths)
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
            subplots_keys = ['figsize', 'facecolor', 'dpi', 'ncols', 'nrows', 'tight_layout']
            config = self.config.filter(subplots_keys, prefix='figure_')
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
        'transpose': (0, 1, 2),
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

    PLOT_DEFAULTS = {
        'save': False,
        'show': True,
        'adjust_figsize': 'image',
    }

    SAVE_DEFAULTS = {
        'bbox_inches': 'tight',
        'pad_inches': 0,
        'save_dpi': 100
    }

    def plot(self, data=None, combine='overlay', mode='image', axes=None, axis=None, ax=None, subplots=None, **kwargs):
        """ Plot data on subplots.

        If a first call and has no subplots, parse axes from kwargs if provided, else create them.
        For every data item choose relevant parameters from config and delegate data plotting to corresponding subplot.
        """
        self.config = PlotConfig({**self.PLOT_DEFAULTS, **self.ANNOTATION_DEFAULTS, **self.SAVE_DEFAULTS, **kwargs})

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
            active_subplots = self.subplots if subplots is None else [self.subplots[idx] for idx in to_list(subplots)]

        mode_defaults = getattr(self, f"{mode.upper()}_DEFAULTS")

        for idx, subplot in enumerate(active_subplots):
            subplot_data = data[idx] if idx < len(data) else None

            if subplot_data is None:
                if subplot.empty:
                    subplot.disable()
                continue

            subplot_idx = None if combine == 'overlay' else idx
            subplot_config = self.config[subplot_idx]
            subplot_config = PlotConfig({**mode_defaults, **subplot_config})

            subplot.plot(mode, subplot_data, subplot_config)

        figure_objects = self.annotate()
        self.figure_objects = figure_objects

        if self.config['adjust_figsize'] is True or self.config['adjust_figsize'] == mode:
            self.adjust_figsize()

        if self.config['show']:
            self.show()
        else:
            self.close()

        if self.config['save'] or 'savepath' in self.config:
            self.save()

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
        text_config = self.config.filter(text_keys, prefix='text_')

        # suptitle
        keys = ['suptitle', 't', 'y']
        suptitle_config = self.config.filter(keys, prefix='suptitle_')
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

    def save(self):
        """ Save plot. """
        default_savepath = datetime.now().strftime('%Y-%m-%d_%H:%M:%S.png')
        savepath = self.config.get('savepath', default_savepath)

        save_keys = ['bbox_inches', 'pad_inches', 'dpi']
        save_config = self.config.filter(save_keys, prefix='save_')

        self.figure.savefig(fname=savepath, **save_config)

    def close(self):
        """ Close figure. """
        plt.close(self.figure)


def plot_image(data, **kwargs):
    """ Shorthand for image plotting. """
    return Plot(data, mode='image', **kwargs)

def plot_histogram(data, **kwargs):
    """ Shorthand for histogram plotting. """
    return Plot(data, mode='histogram', **kwargs)

def plot_curve(data, **kwargs):
    """ Shorthand for curve plotting. """
    return Plot(data, mode='curve', **kwargs)

def plot_loss(data, **kwargs):
    """ Shorthand for loss plotting. """
    return Plot(data, mode='loss', **kwargs)
