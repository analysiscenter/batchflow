""" Plotters. """

try:
    import matplotlib
except ImportError:
    raise ImportError('matplotlib is missing. Install batchflow[image]')
else:
    from .plot import plot
    from .cmaps import *
