""" Plotters. """

try:
    import matplotlib
except ImportError as e:
    raise ImportError('matplotlib is missing. Install batchflow[image]') from e

from .plot import plot
from .cmaps import *
