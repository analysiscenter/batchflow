""" Plot supllementary functions and classes. """
from ast import literal_eval
from colorsys import rgb_to_hls, hls_to_rgb
from numbers import Number
import operator

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.collections import PatchCollection
from matplotlib.colors import ColorConverter, ListedColormap, to_rgba
from matplotlib.patches import Rectangle
from matplotlib.legend_handler import HandlerBase

try:
    from numba import njit
except ImportError:
    from ..decorators import njit

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
        patch = PatchCollection(segments, match_original=True, edgecolor=None, cmap=cmap, label=label)

        return [patch]


class PlotConfig(dict):
    """ Dictionary with additional slicing and filtering capabilities. """

    def maybe_index(self, index=None):
        """ Produce a new config with same keys, but values of list type indexed.

        Parameters
        ----------
        index : int or None
            Index to retrieve from list config values.
            If none provided, get value as it is.
            If value is not a list, do not index it (even if it's an iterable!).

        Raises
        ------
        ValueError
            If parameter is a list but the index is greater than its length.
        """
        result = type(self)()

        if index is None:
            result.update(self)
            return result

        for key, value in self.items():
            if isinstance(value, list):
                try:
                    value = value[index]
                except IndexError as e:
                    msg = f"Tried to obtain element #{index} from `{key}={value}`. Either provide parameter value "\
                            f"as a single item (to use the same `{key}` several times) or add more elements to it."
                    raise ValueError(msg) from e
            result[key] = value

        return result

    def filter(self, keys=None, prefix=None, retrieve='get'):
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
        retrieve : 'get' or 'pop'
            Determines a way desired values are retrieved from config.
        """
        retrieve = getattr(self, retrieve)

        if keys is None and prefix is None:
            raise ValueError("At least `keys` or `prefix` must be specified.")

        if keys is None:
            keys = []

        result = type(self)()

        all_keys = list(self.keys())
        for key in all_keys:
            result_key = None
            if key in keys:
                result_key = key
            elif prefix is not None and key.startswith(prefix):
                result_key = key.split(prefix)[1]

            if result_key is not None:
                result[result_key] = retrieve(key)

        return result

    def update(self, other=None, skip_duplicates=False, **kwargs):
        """ Update config, skipping already present keys if needed. """
        if other is None:
            other = {}

        if skip_duplicates:
            if hasattr(other, 'keys'):
                for key in other.keys():
                    if key not in self:
                        self[key] = other[key]
            else:
                for key, value in other:
                    if key not in self:
                        self[key] = value

            for key, value in kwargs.items():
                if key not in self:
                    self[key] = value

            return type(self)({**other, **kwargs, **self})
        return super().update(other, **kwargs)

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
    for key, operation in STR_TO_OPERATION.items():
        if key in string:
            arg1 = literal_eval(string.split(key)[-1])
            return operation(arg0, arg1)
    msg = f"Given string '{string}' does not contain any of supported operators: {list(STR_TO_OPERATION.keys())}"
    raise ValueError(msg)


@njit()
def is_binary_mask(array):
    """ Fast check that array consists of 0 and 1 only. """
    for item in array:
        if item not in (0., 1.):
            return False
    return True


def contains_numbers(iterable):
    """ Check if first iterable item is a number. """
    return isinstance(iterable[0], Number)


def make_cmap(colors):
    """ Make colormap from provided color/colors list. """
    colors = [ColorConverter().to_rgb(color) if isinstance(color, str) else color for color in colors]
    cmap = ListedColormap(colors)
    return cmap


def extend_cmap(cmap, color, share=0.1, n_points=256, mode='append'):
    """ Make new colormap, adding a new color to existing one.

    Parameters
    ----------
    cmap : valid matplotlib colormap
        Base colormap to extend.
    color : valid matplotlib color
        Color to use for colormap extension.
    share : number from 0 to 1
        New color's share in extended colormap.
    n_points : interger
        Number of points to use for colormap creation.
    mode : 'prepend' or 'append'
        How to extend colormap — from the beginning or from the end.
    """
    if mode == 'append':
        order = slice(None, None, 1)
    elif mode == 'prepend':
        order = slice(None, None, -1)
    else:
        raise ValueError(f"Valid modes are either 'prepend' or 'append', got {mode} instead.")

    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)

    main_points = int(n_points * (1 - share))
    extra_points = n_points - main_points

    if isinstance(color, str):
        color = to_rgba(color)
    elif len(color) == 3:
        color = list(color) + [1]
    color = np.array(color)

    main_colors = cmap(np.linspace(0, 1, main_points))[order]
    step = (color - main_colors[-1]) / extra_points
    steps = np.tile(step, extra_points).reshape(extra_points, 4)
    extra_colors = main_colors[-1] + np.cumsum(steps, axis=0)
    colors = main_colors.tolist() + extra_colors.tolist()
    extended_cmap = make_cmap(colors[order])

    return extended_cmap


def scale_lightness(color, scale):
    """ Make new color with modified lightness from existing. """
    if isinstance(color, str):
        color = ColorConverter.to_rgb(color)
    hue, light, saturation = rgb_to_hls(*color)
    new_color = hls_to_rgb(h=hue, l=min(1, light * scale), s=saturation)
    return new_color


def invert_color(color):
    """ Invert color. """
    return tuple(1 - x for x in to_rgba(color)[:3])


def wrap_by_delimiter(string, width, delimiter=' ', newline='\n'):
    """ Wraps the single paragraph in given `string` allowing breaks at `delimiter` positions only so that every line
        is at most `width` characters long (except for longer indivisible w.r.t. to `delimiter` string items).
    """
    result = ''

    line_len = 0
    line_items = []

    items = string.split(delimiter)
    for item in items:
        item_len = len(item)
        if line_len > 0 and line_len + item_len > width:
            line = delimiter.join(line_items)
            result += line + delimiter + newline
            line_items = []
            line_len = 0
        line_items.append(item)
        line_len += item_len + len(delimiter)

    if line_len > 0:
        line = delimiter.join(line_items)
        result += line

    return result

def ceil_div(a, b):
    """ Return the smallest integer greater than or equal to result of parameters division. """
    return -(-a // b)
