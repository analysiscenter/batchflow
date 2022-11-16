""" Plot primitives. """
from copy import copy
from datetime import datetime
from itertools import cycle
from numbers import Number
from warnings import warn

import numpy as np

from scipy.ndimage import convolve
from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.colors import is_color_like
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, AutoMinorLocator
from mpl_toolkits import axes_grid1

from .utils import CycledList, ColorMappingHandler, PlotConfig
from .utils import evaluate_str_comparison, is_binary_mask, contains_numbers, ceil_div
from .utils import make_cmap, scale_lightness, invert_color, wrap_by_delimiter
from ..utils import to_list



class Layer:
    """ Implements subplot layer, storing plotted object and redirecting call to corresponding matplotlib methods.
        Also handles data transformations in `image` mode, re-applying it to given data on `set_data` call.
    """
    def __init__(self, subplot, mode, index, config, data):
        self.subplot = subplot
        self.mode = mode
        self.index = index
        self.config = config

        # If image is a binary mask, make zeros transparent and set single color for ones
        if mode == 'image' and config['augment_mask'] and is_binary_mask(data.flatten()):
            if self.config.get('mask') is not None:
                msg = f"Since mask augmentation is requested, parameter `mask={self.config['mask']}` is ignored." \
                        " To suppress this warning, provide explicitly `mask=None` " \
                        f"for layer #{self.index} of subplot #{self.subplot.index}."
                warn(msg)
            self.config['mask'] = 0
            if not is_color_like(self.config['cmap']):
                self.config['cmap'] = next(self.subplot.mask_colors)

        preprocessed_data = self.preprocess(data)
        self.objects = getattr(self, mode)(preprocessed_data)

    # Aliases to attributes of parent subplot
    @property
    def ax(self):
        return self.subplot.ax

    @property
    def twin_ax(self):
        return self.subplot.twin_ax

    @property
    def main_object(self):
        return self.objects[0] if hasattr(self, 'objects') else None

    # Preprocessing methods
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
        # pylint: disable=import-outside-toplevel
        import cv2
        dilation_config = self.config.get('dilate', False)

        default_kernel = np.ones((3, 1), dtype=np.uint8)
        if dilation_config:
            if dilation_config is True:
                dilation_config = {'iterations': 1, 'kernel': default_kernel}
            elif isinstance(dilation_config, int):
                dilation_config = {'iterations': dilation_config, 'kernel': default_kernel}
            elif isinstance(dilation_config, tuple):
                dilation_config = {'kernel': np.ones(dilation_config, dtype=np.uint8)}
            elif 'kernel' in dilation_config and isinstance(dilation_config['kernel'], tuple):
                dilation_config['kernel'] = np.ones(dilation_config['kernel'], dtype=np.uint8)
            data = cv2.dilate(data.astype(np.float32), **dilation_config)
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

    def smooth(self, data):
        """ Calculate running average on given data with provided window. """
        window = self.config.get('window')
        if window is not None and window < len(data):
            data = convolve(data, np.ones(window), mode='nearest') / window
            return data
        return None

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
                if hasattr(self, 'objects'):
                    xdata = self.main_object.get_xdata()
                else:
                    xdata = range(len(data))

                data = [xdata, data]

            smoothed = self.smooth(data[1].squeeze() if data[1].ndim > 1 else data[1])
            data = [*data, smoothed]

        if self.mode == 'loss':
            if isinstance(data, tuple):
                loss, lr = data
            else:
                loss, lr = data, None

            if loss is None:
                smoothed = None
            else:
                smoothed = self.smooth(loss)

            data = [loss, smoothed, lr]

        return data

    def update_lims(self):
        """ Recalculate plot limits. """
        self.ax.relim()
        self.ax.autoscale_view()

    def update(self, data):
        """ Preprocess given data and pass it to `set_data`. Does not work in 'histogram' and 'loss' mode. """
        if self.mode == 'image':
            data = self.preprocess(data)
            self.main_object.set_data(data)

            vmin = self.config.get('vmin', np.nanmin(data))
            vmax = self.config.get('vmax', np.nanmax(data))
            self.main_object.set_clim([vmin, vmax])

        if self.mode == 'histogram':
            raise NotImplementedError("Updating layer data is not in supported in 'histogram' mode. ")

        if self.mode == 'curve':
            x_data, y_data = self.preprocess(data)
            self.main_object.set_data(x_data, y_data)
            self.update_lims()

        if self.mode == 'loss':
            raise NotImplementedError("Updating layer data is not in supported in 'loss' mode. ")

    # Plotting methods
    def image(self, data):
        """ Display data as an image. """
        # Assemble colormap from given parameters
        cmap = self.config.get('cmap', None)

        # If a single color provided, prepend 'white' color, so that a resulting tuple defines binary colormap
        if is_color_like(cmap):
            cmap = (cmap, )
        # If a tuple of colors provided in `cmap` argument convert it into a colormap
        if isinstance(cmap, tuple):
            cmap = make_cmap(colors=cmap)
        else:
            cmap = copy(plt.get_cmap(cmap))
        # Set a color for nan/masked values display to colormap if provided
        mask_color = self.config.get('mask_color', None)
        cmap.set_bad(color=mask_color)

        image_keys = ['alpha', 'vmin', 'vmax', 'extent']
        image_config = self.config.filter(keys=image_keys, prefix='image_')
        image = self.ax.imshow(data, cmap=cmap, **image_config)

        return [image]

    def matrix(self, data):
        """ Display data as a matrix. """
        matrix_keys = ['cmap', 'vmin', 'vmax']
        matrix_config = self.config.filter(keys=matrix_keys, prefix='matrix_')

        matrix = self.ax.matshow(data, **matrix_config)

        label_keys = ['size', 'color', 'bbox', 'format']
        label_config = self.config.filter(keys=label_keys, prefix='label_')
        label_format = label_config.pop('format')
        label_color = label_config.pop('color')

        min_value, max_value = np.nanmin(data), np.nanmax(data)
        for y, row in enumerate(data):
            for x, value in enumerate(row):
                normalized_value = (value - min_value) / (max_value - min_value)
                cell_color = plt.get_cmap(matrix_config['cmap'])(normalized_value)
                cell_brightness = np.mean(cell_color[:3])
                color = label_color if cell_brightness < 0.5 else invert_color(label_color)

                if isinstance(label_format, str):
                    formatter = label_format
                elif isinstance(label_format, dict):
                    formatter = ''
                    for dtype, dtype_formatter in label_format.items():
                        if isinstance(value, dtype):
                            formatter = dtype_formatter

                text = format(value, formatter)

                self.subplot.add_text(text=text, x=x, y=y, color=color, **label_config)

        return [matrix]

    def histogram(self, data):
        """ Display data as 1-D histogram. """
        histogram_keys = ['bins', 'color', 'alpha', 'label']
        histogram_config = self.config.filter(histogram_keys, prefix='histogram_')

        _, _, bar = self.ax.hist(data, **histogram_config)

        return [bar]

    def curve(self, data):
        """ Display data as a polygonal chain. """
        x, y, y_smoothed = data

        curve_keys = ['color', 'linestyle', 'alpha', 'label']
        curve_config = self.config.filter(curve_keys, prefix='curve_')

        curves = self.ax.plot(x, y, **curve_config)

        if y_smoothed is not None:
            smoothed_color = scale_lightness(curve_config['color'], scale=.5)
            smoothed_label = self.config.get('smoothed_label')
            _ = self.ax.plot(x, y_smoothed, label=smoothed_label, color=smoothed_color, linestyle='--')

        return curves

    def loss(self, data):
        """ Display a combination of loss curve, its smoothed version and learning rate with nice defaults. """
        loss, smoothed, lr = data

        curves = []

        curve_keys = ['color', 'linestyle', 'linewidth', 'alpha']

        if loss is not None:
            loss_name = self.config.get('label', f"loss #{self.index + 1}")
            loss_label = f'{loss_name} ⟶ {loss[-1]:2.3f}'
            final_window = self.config.get('final_window', None)
            if final_window is not None:
                final_window = min(final_window, len(loss))
                final = np.mean(loss[-final_window:])
                loss_label += f"\nmean over last {final_window} iterations={final:2.3f}"

            loss_config = self.config.filter(keys=curve_keys, prefix='curve_')
            loss_curve = self.ax.plot(loss, label=loss_label, **loss_config)
            curves.extend(loss_curve)

        if smoothed is not None:
            smoothed_color = scale_lightness(loss_config['color'], scale=.5)
            smooth_window = self.config.get('window')
            smoothed_label = self.config.get('smoothed_label', loss_name)
            smoothed_label = smoothed_label + '\n' if smoothed_label else ''
            smoothed_label += f'smoothed with window {smooth_window}'
            smoothed_curve = self.ax.plot(smoothed, label=smoothed_label, color=smoothed_color, linestyle='--')
            curves.extend(smoothed_curve)

        if lr is not None:
            lr_ax = self.ax if loss is None else self.twin_ax
            lr_label = f'learning rate №{self.index + 1} ⟶ {lr[-1]:.0e}'
            lr_config = self.config.filter(keys=curve_keys, prefix='lr_')
            lr_curve = lr_ax.plot(lr, label=lr_label, **lr_config)
            lr_ax.set_ylabel('Learning rate', fontsize=12)
            curves.extend(lr_curve)

        return curves


class Subplot:
    """ Implements plotter subplot, managing new layers creation and axis annotation.
        Basically acts as a proxy over `matplotlib.axes`.
    """
    def __init__(self, plotter, ax, index):
        self.plotter = plotter
        self.ax = ax
        self.index = index
        self._twin_ax = None
        self.layers = []
        self.config = None
        self.annotations = {}

        self.mask_colors = cycle(self.MASK_COLORS)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.layers[key]
        raise ValueError(f"Only integer keys are supported for layers indexing, got {type(key)}.")

    def __call__(self, data, mode='image', **kwargs):
        return self.plot(data=data, mode=mode, **kwargs)

    # Properties
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

    @property
    def lr_ax(self):
        """ Assume that if twin axis of subplot was never created, learning rate values were put on main axis. """
        return self.ax if self._twin_ax is None else self.twin_ax

    @property
    def main_object(self):
        return self.layers[0].main_object if len(self.layers) > 0 else None

    # Subplot default parameters
    COMMON_DEFAULTS = {
        # title
        'title_size': 25,
        'title_pad': 15,
        # axis labels
        'xlabel_size': '12',
        'ylabel_size': '12',
        # grid
        'minor_grid_color': '#CCCCCC',
        'minor_grid_linestyle': '--',
        'major_grid_color': '#CCCCCC',
    }

    # Modes default parameters
    MASK_COLORS = ['mediumseagreen', 'thistle', 'darkorange', 'navy', 'gold', 'firebrick',
                   'red', 'turquoise', 'darkorchid', 'darkkhaki', 'royalblue', 'yellow',
                   'chocolate', 'forestgreen', 'lightpink', 'darkslategray', 'deepskyblue', 'wheat']

    IMAGE_DEFAULTS = {
        # image
        'cmap': 'Greys_r',
        'augment_mask': False,
        'alpha': 1,
        # ticks
        'labeltop': False,
        'labelright': False,
        # values masking
        'mask_color': (0, 0, 0, 0),
        # grid
        'grid': False
    }

    MATRIX_DEFAULTS = {
        # matrix
        'cmap': 'magma',
        'colorbar': True,
        # image axes order
        'transpose': (0, 1, 2),
        # grid
        'grid': False,
        # labels
        'label_size': 12,
        'label_format': {float: '.2f', int: ''},
        'label_bbox': None,
        'label_color': 'white'
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
        'colorbar': None,
        # grid
        'grid': 'major',
        'axisbelow': True
    }

    CURVE_COLORS = ['cornflowerblue', 'sandybrown', 'lightpink', 'mediumseagreen', 'thistle', 'firebrick',
                    'forestgreen', 'navy', 'gold', 'red', 'turquoise', 'darkorchid',
                    'darkkhaki', 'royalblue', 'yellow', 'chocolate', 'darkslategray', 'wheat']

    CURVE_DEFAULTS = {
        # curve
        'color': CycledList(CURVE_COLORS),
        'alpha': 1,
        # axis labels
        'xlabel': 'x',
        'ylabel': 'y',
        # common
        'colorbar': False,
        # grid
        'grid': 'both',
    }

    LOSS_DEFAULTS = {
        # smoothing
        'window': 20,
        'final_window': 50,
        # curve
        'color': CycledList(CURVE_COLORS[::2]),
        'alpha': 1,
        # learning rate
        'lr_color': CycledList(CURVE_COLORS[1::2]),
        # title
        'title': 'Loss values and learning rate',
        # axis labels
        'xlabel': 'Iterations', 'ylabel': 'Loss',
        # common
        'colorbar': False,
        # grid
        'grid': 'both',
        'minor_grid_frequency': (0, 4),
        # legend
        'legend': True,
    }

    @classmethod
    def get_defaults(cls, mode):
        """ Get dictionary with default parameters corresponding to given mode. """
        mode_defaults = getattr(cls, f"{mode.upper()}_DEFAULTS")
        defaults = PlotConfig({**cls.COMMON_DEFAULTS, **mode_defaults})
        return defaults

    # Plot delegator
    def plot(self, data, mode='image', **kwargs):
        """ Update subplot config with given parameters, for every data item create and delegate data plotting to it
        with parameters relevant to this subplot, annotate subplot (add title, labels, colorbar, grid etc.).
        """
        self.config = self.get_defaults(mode)
        self.config.update(kwargs)

        for layer_index, layer_data in enumerate(data):
            layer_config = self.config.maybe_index(layer_index)
            layer = Layer(self, mode=mode, index=layer_index, config=layer_config, data=layer_data)
            self.layers.append(layer)

        annotations = self.annotate(mode)
        self.annotations.update(annotations)

    # Annotation methods
    def annotate(self, mode):
        """ Apply requested annotation functions to subplot with parameters from subplot config. """
        # pylint: disable=too-many-branches
        annotations = {}

        if not self.ax.axison:
            self.enable()

        text_keys = ['fontsize', 'family']
        text_config = self.config.filter(keys=text_keys, prefix='text_')

        # title
        title = self.config.get('title')
        if title is not None:
            if isinstance(title, list):
                title = ', '.join(title)

            title_config = self.config.filter(prefix='title_')
            title_config.update(text_config, skip_duplicates=True)

            title_wrap_config = title_config.filter(prefix='wrap_', retrieve='pop')
            if title_wrap_config:
                title = wrap_by_delimiter(title, **title_wrap_config)

            annotations['title'] = self.ax.set_title(title, **title_config)

        # xlabel
        xlabel = self.config.get('xlabel')
        if xlabel is not None:
            xlabel_config = self.config.filter(prefix='xlabel_')
            xlabel_config.update(text_config, skip_duplicates=True)
            xlabel_wrap_config = xlabel_config.filter(prefix='wrap_', retrieve='pop')
            if xlabel_wrap_config:
                xlabel = wrap_by_delimiter(xlabel, **xlabel_wrap_config)
            annotations['xlabel'] = self.ax.set_xlabel(xlabel=xlabel, **xlabel_config)

        # ylabel
        ylabel = self.config.get('ylabel')
        if ylabel is not None:
            ylabel_config = self.config.filter(prefix='ylabel_')
            ylabel_config.update(text_config, skip_duplicates=True)
            ylabel_wrap_config = ylabel_config.filter(prefix='wrap_', retrieve='pop')
            if ylabel_wrap_config:
                ylabel = wrap_by_delimiter(ylabel, **ylabel_wrap_config)
            annotations['ylabel'] = self.ax.set_ylabel(ylabel=ylabel, **ylabel_config)

        # xticks
        xtick_config = self.config.filter(prefix='xtick_')
        if 'locations' in xtick_config:
            xtick_locations = xtick_config.pop('locations')
            self.ax.set_xticks(xtick_locations)

        if xtick_config:
            if 'labels' not in xtick_config:
                xtick_config['labels'] = [item.get_text() for item in self.ax.get_xticklabels()]
            if 'size' in xtick_config:
                xtick_config['fontsize'] = xtick_config.pop('size')
            annotations['xticks'] = self.ax.set_xticklabels(**xtick_config)

        # yticks
        ytick_config = self.config.filter(prefix='ytick_')
        if 'locations' in ytick_config:
            xtick_locations = ytick_config.pop('locations')
            self.ax.set_yticks(xtick_locations)

        if ytick_config:
            if 'labels' not in ytick_config:
                ytick_config['labels'] = [item.get_text() for item in self.ax.get_yticklabels()]
            if 'size' in ytick_config:
                ytick_config['fontsize'] = ytick_config.pop('size')
            annotations['xticks'] = self.ax.set_yticklabels(**ytick_config)

        # ticks
        tick_keys = ['labeltop', 'labelright', 'labelcolor']
        tick_config = self.config.filter(keys=tick_keys, prefix='tick_')
        if tick_config:
            self.ax.tick_params(**tick_config)

        # Change scale of axis, if needed
        if self.config.get('log') or self.config.get('log_loss'):
            self.ax.set_yscale('log')

        if self.config.get('log_lr'):
            self.lr_ax.set_yscale('log')

        # Change x-axis limits
        xlim = self.config.get('xlim')
        self.ax.set_xlim(xlim)

        # Change y-axis limits
        ylim = self.config.get('ylim')
        self.ax.set_ylim(ylim)

        # Add image colorbar
        colorbar = self.config.get('colorbar')
        if colorbar is not None and self.main_object is not None:
            colorbar_config = self.config.filter(prefix='colorbar_')

            colorbar_config['fake'] = not colorbar

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

            annotations['colorbar'] = self.add_colorbar(image=self.main_object, **colorbar_config)

        # Add legend
        if mode in ('loss', 'curve'):
            self.config['label'] = [obj for layer in self.layers for obj in layer.objects]

        label = self.config.get('label')
        if label is not None:
            if not isinstance(label, list):
                label = [label]

            legend_config = self.config.filter(prefix='legend_')

            if 'color' not in legend_config:
                color_key = 'cmap' if mode in ('image', 'matrix') else 'color'
                legend_config['color'] = [layer.config[color_key] for layer in self]

            if 'alpha' not in legend_config:
                legend_config['alpha'] = [layer.config['alpha'] for layer in self]

            annotations['legend'] = self.add_legend(mode=mode, label=label, **legend_config)

        # Add grid
        grid = self.config.get('grid', None)
        grid_config = self.config.filter(prefix='grid_')

        if grid in (None, False):
            self.ax.grid(False)

        if grid in ('minor', 'both'):
            minor_grid_config = self.config.filter(prefix='minor_grid_')
            minor_grid_config = minor_grid_config.update(grid_config, skip_duplicates=True)
            self.add_grid(grid_type='minor', **minor_grid_config)

        if grid in ('major', 'both'):
            major_grid_config = self.config.filter(prefix='major_grid_')
            major_grid_config = major_grid_config.update(grid_config, skip_duplicates=True)
            self.add_grid(grid_type='major', **major_grid_config)

        # Set whether axis ticks and gridlines are above or below most artists.
        self.ax.set_axisbelow(self.config.get('axisbelow', False))

        # Change facecolor
        facecolor = self.config.get('facecolor', None)
        if facecolor is not None:
            self.ax.set_facecolor(facecolor)

        return annotations

    def add_colorbar(self, image, width=0.2, pad=None, color='black', position='right', fake=False, **kwargs):
        """ Append colorbar to the subplot on the right. """
        divider = axes_grid1.make_axes_locatable(image.axes)
        cax = divider.append_axes(position=position, size=width, pad=pad)

        if fake:
            cax.set_axis_off()
            return None

        colorbar = image.axes.figure.colorbar(image, cax=cax, **kwargs)
        colorbar.ax.yaxis.set_tick_params(color=color, labelcolor=color)

        return colorbar

    def add_legend(self, mode='image', label=None, color='none', alpha=1,
                   size=15, family='sans-serif', properties=None, **kwargs):
        """ Add patches to subplot legend.

        Parameters
        ----------
        mode : 'image', 'histogram', 'curve', 'loss'
            Mode to match legend hadles patches to.
            If from ('image', 'histogram'), use rectangular legend patches.
            If from ('curve', 'loss'), use line legend patches.
        label : list of str and Artist
            If str, a text to show next to the legend patch/line.
            If Artist, must be valid handle for `plt.legend` (line, patch etc.)
        color : list of str, matplotlib colormap object or tuples of str
            Colors to use for patches creation if those are not provided explicitly.
            Must be valid matplotlib color (e.g. 'salmon', '#120FA3', (0.3, 0.4, 0.5)).
        alpha : number or list of numbers from 0 to 1
            Legend handles opacity.
        size : int
            Legend text font size.
        family : str
            Legent text font family.
        properties : dict
            Legend font parameters, must be valid for `matplotlib.font_manager.FontProperties`.
        kwargs : misc
            For `matplotlib.legend`.
        """
        if properties is None:
            properties = {}
        properties = {'size': size, 'family': family, **properties}

        # get legend that already exists
        legend = self.ax.get_legend()
        old_handles = getattr(legend, 'legendHandles', [])
        handler_map = getattr(legend, '_custom_handler_map', {})

        # make new handles
        new_handles = []
        labels = to_list(label)
        colors = [color] * len(labels) if isinstance(color, str) else color
        alphas = [alpha] * len(labels) if isinstance(alpha, Number) else alpha

        for label_item, label_color, label_alpha in zip(labels, colors, alphas):
            if label_item is None:
                continue

            if isinstance(label_item, str):
                if mode in ('image', 'histogram'):
                    if is_color_like(label_color):
                        handle = Patch(color=label_color, alpha=label_alpha, label=label_item)
                    else:
                        handle = PatchCollection(patches=[], cmap=label_color, label=label_item)
                        handler_map[PatchCollection] = ColorMappingHandler()
                elif mode in ('curve', 'loss'):
                    handle = Line2D(xdata=[0], ydata=[0], color=label_color, alpha=label_alpha, label=label_item)
                new_handles.append(handle)
            elif not label_item.get_label().startswith('_'):
                new_handles.append(label_item)

        if len(new_handles) > 0:
            # extend existing handles and labels with new ones
            handles = old_handles + new_handles
            legend = self.ax.legend(prop=properties, handles=handles,  handler_map=handler_map, **kwargs)

        return legend

    def add_text(self, text, size=10, x=0.5, y=0.5, ha='center', va='center', bbox='default', **kwargs):
        """ Add text to subplot.

        A convenient method for adding text in box (usually on empty subplot).

        Parameters
        ----------
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

    def add_grid(self, grid_type, frequency=None, zorder=0, **kwargs):
        """ Set parameters for subplot axis grid. """
        if grid_type == 'minor':
            locator = AutoMinorLocator
        elif grid_type == 'major':
            locator = MaxNLocator

        if isinstance(frequency, tuple):
            x_frequency, y_frequency = frequency
        else:
            x_frequency, y_frequency = (frequency, frequency)

        if x_frequency:
            set_locator = getattr(self.ax.xaxis, f'set_{grid_type}_locator')
            set_locator(locator(x_frequency))

        if y_frequency:
            set_locator = getattr(self.ax.yaxis, f'set_{grid_type}_locator')
            set_locator(locator(y_frequency))

        self.ax.grid(which=grid_type, zorder=zorder, **kwargs)

    def enable(self):
        """ Make subplot visible. """
        self.ax.set_axis_on()

    def disable(self):
        """ Make subplot invisible. """
        self.ax.set_axis_off()

    def clear(self):
        """ Clear subplot axis. """
        colorbar = self.annotations.get('colorbar')
        if colorbar is not None:
            colorbar.remove()

        self.annotations = {}

        self.ax.clear()
        for layer in self.layers:
            for obj in layer.objects:
                obj.remove()
        self.layers = []


class Plot:
    """ Multiple graphs plotter.

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

    Notes
    -----
    • Data items combination.

    1. The simplest scenario if when a single data array provided — in that case data is displayed on a single subplot:
       >>> plot(array)
    2. A more advanced use case is when one provide a list of arrays — plot behaviour depends on `combine` parameter:
       a. Images are put on same subplot and overlaid one over another if `combine='overlay'` (which is default):
          >>> plot([image_0, mask_0])
       b. Images are put on separate subplots if `combine='separate'`.
          >>> plot([image_0, image_1], combine='separate')
    3. The most complex scenario is displaying images in a 'mixed' manner (ones overlaid and others separated).
       For example, to overlay first two images but to display the third one separately, use the following notation:
       >>> plot([[image_0, mask_0], image_1]); (`combine='mixed'` is set automatically if data is double-nested).

    The order of arrays inside the double-nested structure basically declares, which of them belong to the same subplot
    and therefore should be rendered one over another, and which must be displayed separately.


    • Parameters nestedness.

    If a parameter is provided in a list, each subplot uses its item on position corresponding to its index and
    every subplot layer in turn uses item from that sublist on positions that correspond to its index w.r.t. to subplot.
    Therefore, such parameters must resemble data nestedness level, since that allows binding subplots and parameters.
    However, it's possible for parameter to be a single item — in that case it's shared across all subplots and layers.

    For example, to display two images separately with same colormap, the following code required:
    >>> plot([image_0, image_1], combine='separate', cmap='viridis')
    If one wish to use different colormaps for every image, the code should be like this:
    >>> plot([image_0, image_1], combine='separate', cmap=['viridis', 'magma'])
    Finally, if a more complex data provided, the parameter nestedness level must resemble the one in data:
    >>> plot([[image_0, mask_0], [image_1, mask_1]], cmap=[['viridis', 'red'], ['magma', 'green']])


    • Advanced parameters managing.

    Keep in mind, that set of parameters that are parsed by plotter directly is limited to ones most frequently used.
    However there is a way to provide any parameter to a specific plot method, using prefix corresponding to it.
    One must prepend that prefix to a parameter name itself and provide parameter value in argument under such name.

    This also allows one to pass arguments of the same name for different plotting steps.
    E.g. `plt.set_title` and `plt.set_xlabel` both require `size` argument.
    Providing `{'size': 30}` in kwargs will affect both title and x-axis labels.
    To change parameter for title only, one can provide {'title_fontsize': 30}` instead.

    See specific prefices examples in sections below.

    Parameters
    ----------
    • General:

    data : np.ndarray, tuple or list
        If array, its dimensionality must match plot `mode`:
        - in 'image' mode 1d, 2d and 3d arrays are valid, thoug 3d image must be either 1- or 3- channeled;
        - in 'histogram' mode arrays of any dimensionality are valid, since they are flattened anyway;
        - in 'curve' and 'loss' modes 1d arrays are valid, defining polyline 'y' coordinates;
        If tuple, must contain two 1d arrays:
        - in 'curve' mode arrays define 'x' and 'y' polyline coordinates correspondingly;
        - in 'loss' mode array define 'y' coordinates of loss and learning rates popylines correspondingly;
        If list, must either contain arrays or tuples of format specified above. List might be either flat or nested.
        If list if flat, plotter parses data based on `combine` parameter value (see details below).
        If list is nested, outer level defines subplots order while inner one defines layers order.
    mode : 'image', 'histogram', 'curve', 'loss'
        If 'image' plot given arrays as images.
        If 'histogram' plot 1d histogram.
        If 'curve' plot given arrays as curve lines.
        If 'loss' plot given arrays as loss curves.
    combine : 'overlay', 'separate' or 'mixed'
        Whether overlay images on a single subplot, show them on separate ones or use mixed approach.
        Needs specifying only when `combine='separate'` required, since `combine='overlay'` is default and
        `combine='mixed'` is infered automatically from data (if data list is nested, no need specifiying `combine`).
    fix_config : bool
        If False, every time `plot` is called update config with provided keyword arguments, replacing older parameters.
        If True, fix plotter config as provided on initialization. Usefull, if one want to reuse this config on updates.
    kwargs :
        - For one of `image`, `histogram`, `curve`, `loss` methods of `Layer` (depending on chosen mode).
            Parameters and data nestedness levels must match if they are lists meant for differents subplots/layers.
            Every param with 'image_', 'histogram_', 'curve_', 'loss_' prefix is redirected to corresponding method.
            See detailed parameters listings below.
        - For `annotate`.
            Every param with 'title_', 'suptitle_', 'xlabel_', 'ylabel_', 'xticks_', 'yticks_', 'xlim_', 'ylim_',
            colorbar_', 'legend_' or 'grid_' prefix is redirected to corresponding matplotlib method.
            Also 'facecolor', 'set_axisbelow', 'disable_axes' arguments are accepted.


    • Figure:

    figsize : tuple
        Size of displayed figure. If not provided, infered from data shapes.
    facecolor : string or tuple of 3 or 4 numbers
        Figure background color. Must be valid matplotlib color (e.g. 'salmon', '#120FA3', (0.3, 0.4, 0.5)).
    dpi : float
        The resolution of the figure in dots-per-inch.
    ncols, nrows : int
        Number of figure columns/rows.
    tight_layout : bool
        Whether adjust subplot parameters using `plt.tight_layout` with default padding or not. Defaults is True.
    sharex, sharey : bool
        Whether to use same x/y-axis for all subplots.
    figure_{parameter} : misc
        Any parameter valid for `plt.subplots`. For example, `figure_gridspec_kw=True`.


    • Image:

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
        Any parameter valid for `Axes.imshow`. For example, `image_interpolate='bessel'`.


    • Histogram:

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
        Any parameter valid for `Axes.hist`. For example, `histogram_density=True`.


    • Curve:

    color : string or tuple of 3 or 4 numbers
        Color to display curve with. Must be valid matplotlib color (e.g. 'salmon', '#120FA3', (0.3, 0.4, 0.5)).
    linestyle : str
        Style to display curve with. Must be valid matplotlib line style (e.g. 'dashed', ':').
    alpha : number in (0, 1) range
        Curve opacity (0 means fully transparent, i.e. invisible, 1 - totally opaque). Useful when `combine='overlay'`.
    curve_{parameter} : misc
        Any parameter valid for `Axes.plot`. For example, `curve_marker='h'`.


    • Loss:

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
        Any parameter valid for `Axes.plot`. For example, `curve_fillstyle='bottom'`.


    • Text:

    {text_object}_color : str, matplotlib colormap object or tuple
        Color of corresponding text object. Valid objects are 'suptitle', 'title', 'xlabel', 'ylabel', 'legend'.
        If str, must be valid matplotlib colormap.
        Must be valid matplotlib color (e.g. 'roaylblue', '#120FA3', (0.3, 0.4, 0.5)).
    {text_object}_size : number
        Size of corresponding text object. Valid objects are 'suptitle', 'title', 'xlabel', 'ylabel', 'legend'.


    • Suptitle:

    suptitle : str
        Text of suptitle label.
    suptitle_{parameter} : misc
        Any parameter valid for `Figure.suptitle`. For example, `suptitle_y=1.05`.


    • Title:

    title : str or list of str
        Text of title label. If a list of string, items are joined in a single string with words separated by commas.
    title_{parameter} : misc
        Any parameter valid for `Axes.set_title`. For example, `title_loc='left'`.


    • Axes labels:

    xlabel : str
        Text of x-axis label.
    xlabel_{parameter} : misc
        Any parameter valid for `Axes.set_xlabel`. For example `xlabel_labelpad=0.1`.
    ylabel : str
        Text of y-axis label.
    ylabel_{parameter} : misc
        Any parameter valid for `Axes.set_ylabel`. For example `ylabel_labelpad=0.1`.


    • Axes ticks:

    xtick_locations : list of numbers
        Positions of x-axis ticks.
    xtick_{parameter} : misc
        Any parameter valid for `Axes.set_xticklabels`. For example `xtick_labels=['Background', 'Class 0', 'Class 1']`.
    ytick_locations : list of numbers
        Positions of y-axis ticks.
    ytick_{parameter} : misc
        Any parameter valid for `Axes.set_yticklabels`. For example `ytick_labels=['Background', 'Class 0', 'Class 1']`.
    tick_{parameter} : misc
        Any parameter valid for `Axes.tick_params`. For example `tick_labelbottom=False`.


    • Axes scaling:

    log : bool
        If True, set scale of y-axis to logarithmic.
    log_loss, log_lr : bool
        If True, set scale of y-axis corresponding to loss/learning rate to logarithmic.


    • Axes limits:

    xlim : number or tuple of two numbers
        If a single number, defines left limit of x-axis. If a tuple, defines both left and right x-axis limits.
    ylim : number or tuple of two numbers
        If a single number, defines left limit of y-axis. If a tuple, defines both left and right y-axis limits.


    • Colorbar:

    colorbar : None or bool
        If None no colorbar added and subplot axis left unchanged.
        If False the place for colorbar is reserved on subplot axis to right to the main object but left empty.
        If True the place for colorbar is reserved on subplot axis to right to the main object and colorbar is added.
    colorbar_width : number
        Thickness of colorbar object.
    colorbar_pad : None or number
        Distance between colorbar object and main axis object. If None, calculated automatically.
    colorbar_{parameter} : misc
        Any parameter valid for `Figure.colorbar`. For example `colorbar_label='Values range'`.


    • Legend:

    label : None, str or `matplotlib.Artist`
        If str, a text to show next to the legend patches/lines corresponding to layers main objects.
        If Artist, must be valid handle for `plt.legend` (patch, line etc.)
    legend_color : None, str or matplotlib colormap object
        Color of legend handles. If None, color of corresponding main subplot object is used.
        If str, must be valid matplotlib colormap or color (e.g. 'ocean', 'roaylblue', '#120FA3', (0.3, 0.4, 0.5)).
    legend_alpha : None or number
        Opacity of legend handles, must be from [0, 1] range. If None, opacity of main subplot object is used.
    legend_size : number
        Size of legend handles.
    legend_{parameter} : misc
        Any parameter valid for `plt.legend`. For example `legend_loc='center'`.


    • Grid:

    grid: 'minor', 'major' or 'both'
        Grid type to show.
    grid_frequency : number or tuple of two numbers
        If a single number, defines grid frequency for both subplot axes.
        If a tuple of two numbers, they define grid frequencies for x-axis and y-axis correspondingly.
    grid_{parameter} : misc
        Any parameter valid for `Axes.grid`. For example `grid_color='tan'`.
    minor_grid_{parameter} : misc
        Same as above, but applied to minor grid only.
    major_grid_{parameter} : misc
        Same as above, but applied to major grid only.
    axisbelow : bool
        Set whether axis ticks and gridlines are above or below most artists.
    """
    MODES = ['image', 'matrix', 'histogram', 'curve', 'loss']

    def __init__(self, data=None, mode='image', combine='overlay', fix_config=False, **kwargs):
        self.figure = None
        self.subplots = None
        self.config = self.get_defaults(mode)
        self.figure_config = None
        self.fix_config = fix_config

        self.fresh = True
        self.plot(data=data, combine=combine, mode=mode, **kwargs)
        self.fresh = False

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.subplots[key]
        raise ValueError(f"Only integer keys are supported for subplots indexing, got {type(key)}.")

    def __call__(self, data, combine='overlay', mode='image', **kwargs):
        return self.plot(data, combine=combine, mode=mode, **kwargs)

    def __repr__(self):
        return f"<Plotter with {len(self.subplots)} subplots>"

    def __str__(self):
        return f"<Plotter with {len(self.subplots)} subplots>"

    def _ipython_display_(self):
        return None

    # Data conversion methods
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
        data_list = []

        if data is None:
            data_list = []
        elif isinstance(data, tuple):
            data_list = [[cls.parse_tuple(data=data, mode=mode)]]
        elif isinstance(data, np.ndarray):
            data_list = [[cls.parse_array(data=data, mode=mode)]]
        elif isinstance(data, list) and contains_numbers(data):
            data_list = [[np.array(data)]]
        elif isinstance(data, list):
            if any(isinstance(item, list) and not contains_numbers(item) for item in data):
                combine = 'mixed'

            data_list = []
            for item in data:

                if item is None:
                    if combine == 'overlay':
                        msg = "`None` is a future subplots placeholder. It makes not sense when `combine='overlay'`."
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
                        raise ValueError("Data list items can't be lists themselves when `combine='separate'`")
                    data_item = []
                    for subitem in item:
                        if isinstance(subitem, tuple):
                            data_item += [cls.parse_tuple(data=subitem, mode=mode)]
                        elif isinstance(subitem, np.ndarray):
                            data_item += [cls.parse_array(data=subitem, mode=mode)]
                        elif isinstance(subitem, list) and contains_numbers(subitem):
                            data_item += [np.array(subitem)]
                        elif isinstance(subitem, list):
                            msg = f"Valid data items are None, tuple, array or list of those, got {type(item)}."
                            raise ValueError(msg)
                else:
                    msg = f"Valid data items are None, tuple, array or list of those, got {type(item)}."
                    raise ValueError(msg)

                if combine == 'overlay':
                    data_list.extend(data_item)
                elif combine in ('separate', 'mixed'):
                    data_list.append(data_item)
                else:
                    msg = f"Valid combine modes are 'overlay', 'separate', 'mixed', got {combine}."
                    raise ValueError(msg)

            if combine == 'overlay':
                data_list = [data_list]

        return data_list, combine

    # Figure manipulation methods
    @staticmethod
    def infer_ncols_nrows(n_subplots, ncols, nrows, max_ncols, **kwargs):
        """ Infer number of figure columns and rows from number of provided data items. """
        _ = kwargs

        # Make ncols/nrows
        if ncols is None and nrows is None:
            ncols = min(max_ncols, n_subplots)
            nrows = ceil_div(n_subplots, ncols)
        elif ncols is None:
            ncols = ceil_div(n_subplots, nrows)
        elif nrows is None:
            nrows = ceil_div(n_subplots, ncols)

        return ncols, nrows

    @staticmethod
    def infer_figure_ratio(mode, n_subplots, data, ncols, nrows, xlim, ylim, transpose):
        """ Infer default figure height/width ratio from shapes of provided data. """
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

                min_height = 0 if ylim[idx][0] is None else ylim[idx][0]
                max_height = shape[transpose[0]] if ylim[idx][1] is None else ylim[idx][1]
                subplot_height = abs(max_height - min_height)
                heights.append(subplot_height)

                min_width = shape[transpose[1]] if xlim[idx][0] is None else xlim[idx][0]
                max_width = 0 if xlim[idx][1] is None else xlim[idx][1]
                subplot_width = abs(max_width - min_width)
                widths.append(subplot_width)

            mean_height, mean_width = np.mean(heights), np.mean(widths)
            if np.isnan(mean_height) or np.isnan(mean_width):
                ratio = 1
            else:
                ratio = (mean_height * nrows) / (mean_width * ncols)

        elif mode == 'matrix':
            ratio = 1

        elif mode == 'histogram':
            ratio = 2 / 3 / ncols * nrows

        elif mode in ('curve', 'loss'):
            ratio = 1 / 3 / ncols * nrows

        return ratio

    @classmethod
    def infer_figure_size(cls, mode, n_subplots, data, ncols, nrows, ratio, scale,
                          max_fig_width, xlim, ylim, transpose, subplot_width, **kwargs):
        """ Infer default figure size from shapes of provided data. """
        _ = kwargs

        if ratio is None:
            ratio = cls.infer_figure_ratio(mode, n_subplots, data, ncols, nrows, xlim, ylim, transpose)

        fig_width = subplot_width * ncols * scale
        fig_height = fig_width * ratio

        if fig_width > max_fig_width:
            fig_width = max_fig_width
            fig_height = fig_width * ratio

        figsize = (fig_width, fig_height)
        return figsize

    def make_subplots(self, mode, n_subplots, data, axes=None, figure=None):
        """ Create figure and axes if needed. """
        if axes is None and figure is not None:
            axes = figure.axes

        if axes is None:
            if self.config['ncols'] is None or self.config['nrows'] is None:
                self.config['ncols'], self.config['nrows'] = self.infer_ncols_nrows(n_subplots, **self.config)

            if self.config['figsize'] is None:
                self.config['figsize'] = self.infer_figure_size(mode, n_subplots, data, **self.config)

            figure_keys = ['figsize', 'ncols', 'nrows', 'facecolor', 'dpi', 'tight_layout', 'sharex', 'sharey']
            figure_config = self.config.filter(keys=figure_keys, prefix='figure_')
            figure, axes = plt.subplots(**figure_config)
            axes = to_list(axes)
        else:
            axes = to_list(axes)
            if len(axes) < n_subplots:
                raise ValueError(f"Not enough axes provided — got ({len(axes)}) for {n_subplots} subplots.")

            figure = axes[0].figure
            ncols, nrows = figure.axes[0].get_subplotspec().get_gridspec().get_geometry()
            figure_config = {
                'ncols': ncols,
                'nrows': nrows,
                'figsize': figure.get_size_inches(),
                'dpi': figure.dpi
            }

        subplots = [Subplot(self, ax=ax, index=ax_num) for ax_num, ax in enumerate(axes)]

        return figure, subplots, figure_config

    def get_bbox(self, obj):
        """ Get object bounding box in inches. """
        renderer = self.figure.canvas.get_renderer()
        transformer = self.figure.dpi_scale_trans.inverted()
        return obj.get_window_extent(renderer=renderer).transformed(transformer)

    def adjust_figsize(self):
        """ Look through subplots annotation objects and add figsize corrections for their widths and heights. """
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
                if len(yticks_objects) > 0:
                    first_ytick_bbox = self.get_bbox(yticks_objects[0]) # first lower xticklabel bbox
                    lower_yticks_width = max(0, ax_bbox.x0 - first_ytick_bbox.x0)
                    width += lower_yticks_width

                if len(yticks_objects) > 1:
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

        nrows, ncols = self.figure_config['ncols'], self.figure_config['nrows']
        ax_widths = np.array(ax_widths).reshape(nrows, ncols)
        extra_width += ax_widths.max(axis=1).sum()

        ax_heights = np.array(ax_heights).reshape(nrows, ncols)
        extra_height += ax_heights.max(axis=0).sum()

        fig_width, fig_height = self.figure_config['figsize']
        new_figsize = (fig_width + extra_width, fig_height + extra_height)
        self.figure.set_size_inches(new_figsize)

    # Figure rendering and saving defaults
    COMMON_DEFAULTS = {
        # general
        'facecolor': 'white',
        'text_color' : 'black',
        # figure
        'tight_layout': True,
        'ncols': None, 'nrows': None, # infer from data
        'figsize': None, 'ratio': None, # infer from data
        'scale': 1, 'transpose': None,
        'max_fig_width': 25,
        'xlim': (None, None),
        'ylim': (None, None),
        # suptitle
        'suptitle_size': 20,
        # save
        'bbox_inches': 'tight',
        'pad_inches': 0,
        'save_dpi': 100,
    }

    IMAGE_DEFAULTS = {
        'max_ncols': 4,
        'subplot_width': 8,
        'transpose': (0, 1, 2)
    }

    MATRIX_DEFAULTS = {
        'max_ncols': 4,
        'subplot_width': 8
    }

    HISTOGRAM_DEFAULTS = {
        'max_ncols': 4,
        'subplot_width': 8
    }

    CURVE_DEFAULTS = {
        'max_ncols': 1,
        'subplot_width': 16
    }

    LOSS_DEFAULTS = {
        'max_ncols': 1,
        'subplot_width': 16
    }

    @classmethod
    def get_defaults(cls, mode):
        """ Get dictionary with default parameters corresponding to given mode. """
        mode_defaults = getattr(cls, f"{mode.upper()}_DEFAULTS")
        defaults = PlotConfig({**cls.COMMON_DEFAULTS, **mode_defaults})
        return defaults

    # Plotting delegator
    def plot(self, data=None, combine='overlay', mode='image', show=True, force_show=False, save=False,
             axes=None, positions=None, n_subplots=None, adjust_figsize='image', **kwargs):
        """ Plot data on subplots.

        If a first call (from `__init__`), parse axes from kwargs if they are provided, else create them.
        For every data item choose relevant parameters from config and delegate data plotting to corresponding subplot.
        """
        if mode not in self.MODES:
            raise ValueError(f"Unknown mode '{mode}'. Expected one of {self.MODES}.")

        data, combine = self.parse_data(data=data, combine=combine, mode=mode)

        if n_subplots is None:
            n_subplots = 1 if combine == 'overlay' else len(data)

        if self.fresh:
            self.config.update(kwargs)
            figure, subplots, figure_config = self.make_subplots(mode=mode, n_subplots=n_subplots, data=data, axes=axes)
            self.figure, self.subplots, self.figure_config = figure, subplots, figure_config
        else:
            if not self.fix_config:
                self.config = PlotConfig()
            outer_config = PlotConfig(kwargs)
            if axes is not None:
                msg = "Subplots already created and new axes cannot bespecified."
                raise ValueError(msg)

        positions = list(range(len(data))) if positions is None else to_list(positions)
        for absolute_index, subplot in enumerate(self.subplots):
            relative_index = positions.index(absolute_index) if absolute_index in positions else None
            subplot_data = None if relative_index is None else data[relative_index]

            if subplot_data is None:
                if subplot.main_object is None:
                    subplot.disable()
                continue

            subplot_index = None if combine == 'overlay' else relative_index
            subplot_config = self.config.maybe_index(subplot_index)

            if not self.fresh:
                outer_subplot_config = outer_config.maybe_index(subplot_index)
                subplot_config.update(outer_subplot_config)

            subplot.plot(data=subplot_data, mode=mode, **subplot_config)

        self.figure_objects = self.annotate()

        if adjust_figsize is True or adjust_figsize == mode:
            self.adjust_figsize()

        if force_show:
            self.force_show()

        if not show:
            self.close()

        if save or 'savepath' in self.config:
            self.save()

        return self

    def annotate(self):
        """ Put suptitle with given parameters over figure and apply `tight_layout`. """
        annotations = {}

        text_keys = ['fontsize', 'family']
        text_config = self.config.filter(keys=text_keys, prefix='text_')

        # suptitle
        suptitle = self.config.get('suptitle')
        if suptitle is not None:
            suptitle_config = self.config.filter(prefix='suptitle_')
            suptitle_config.update(text_config, skip_duplicates=True)

            suptitle_wrap_config = suptitle_config.filter(prefix='wrap_')
            if suptitle_wrap_config:
                suptitle = wrap_by_delimiter(suptitle, **suptitle_wrap_config)

            annotations['suptitle'] = self.figure.suptitle(suptitle, **suptitle_config)

        self.figure.tight_layout()

        return annotations

    # Result finalizing methods
    @staticmethod
    def force_show():
        plt.show()

    def redraw(self):
        """ Draw figure again by creating dummy figure and using its manager to display original figure. """
        dummy_figure = plt.figure()
        new_manager = dummy_figure.canvas.manager
        new_manager.canvas.figure = self.figure
        self.figure.set_canvas(new_manager.canvas)
        plt.show(block=False)

    def clear(self):
        self.figure.clear()

    def clear_subplots(self):
        for subplot in self.subplots:
            subplot.clear()

    def save(self, **kwargs):
        """ Save plot. """
        save_keys = ['savepath', 'bbox_inches', 'pad_inches', 'dpi']
        save_config = self.config.filter(keys=save_keys, prefix='save_')
        save_config.update(kwargs)

        default_savepath = datetime.now().strftime('%Y-%m-%d_%H:%M:%S.png')
        savepath = save_config.pop('savepath', default_savepath)

        self.figure.savefig(fname=savepath, **save_config)

    def close(self):
        """ Close figure. """
        plt.close(self.figure)

plot = Plot # an alias
