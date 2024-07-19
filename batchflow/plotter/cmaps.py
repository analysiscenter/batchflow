""" Custom colormaps. """
from matplotlib import colormaps
from .utils import extend_cmap



BATCHFLOW_CMAP = extend_cmap('magma', 'white')
try:
    colormaps.register(cmap=BATCHFLOW_CMAP, name='batchflow')
except ValueError:
    pass
