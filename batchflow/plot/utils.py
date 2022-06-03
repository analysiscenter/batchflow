""" Plot supllementary functions and classes. """
from ast import literal_eval
from colorsys import rgb_to_hls, hls_to_rgb
from numbers import Number
import operator

from matplotlib.collections import PatchCollection
from matplotlib.colors import ColorConverter, ListedColormap
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
        patch = PatchCollection(segments, match_original=True, edgecolor=None, cmap=cmap.name, label=label)

        return [patch]


class PlotConfig(dict):
    """ Dictionary with additional filtering capabilities. """

    def maybe_index(self, key, index):
        """ Get i-th element of parameter if index is provided and parameter value is a list else return it unchanged.

        Parameters
        ----------
        key : str
            Parameter name.
        index : int or None
            Index to retrieve from value stored under provided key.
            If none provided, get argument value as it is.
            If value is not a list, do not index it (even if it's an iterable!).

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

    def filter(self, keys=None, prefix='', index=None):
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
        index : int or None
            Index to retrieve from value stored under provided key.
            If none provided, get argument value as it is.
            If value is not a list, do not index it (even if it's an iterable!).
        """
        if keys is None:
            keys = list(self.keys())
        elif prefix:
            keys += [key.split(prefix)[1] for key in self if key.startswith(prefix)]

        result = type(self)()

        for key in keys:
            if prefix + key in self:
                result[key] = self.maybe_index(prefix + key, index)
            elif key in self:
                result[key] = self.maybe_index(key, index)

        return result


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


@njit()
def is_binary(array):
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


def scale_lightness(color, scale):
    """ Make new color with modified lightness from existing. """
    if isinstance(color, str):
        color = ColorConverter.to_rgb(color)
    hue, light, saturation = rgb_to_hls(*color)
    new_color = hls_to_rgb(h=hue, l=min(1, light * scale), s=saturation)
    return new_color
