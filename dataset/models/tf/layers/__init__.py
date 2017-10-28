""" Custom tf layers and operations """
import numpy as np
import tensorflow as tf

from .core import flatten, flatten2d, maxout, mip
from .conv import conv_block, conv1d_block, conv2d_block, conv3d_block
from .conv1d_tr import conv1d_transpose
from .pooling import max_pooling, average_pooling
