""" Custom colormaps. """
from matplotlib.pyplot import register_cmap
from .utils import extend_cmap



BATCHFLOW_CMAP = extend_cmap('magma', 'white')
try:
    register_cmap('batchflow', BATCHFLOW_CMAP)
except ValueError:
    pass
