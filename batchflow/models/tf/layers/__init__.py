""" Custom tf layers and operations """
#pylint: disable=no-name-in-module
from .layer import Layer
from .conv_block import conv_block, ConvBlock, update_layers
from .core import flatten, flatten2d, activation, dense, maxout, mip, xip, combine
from .core import Flatten, Flatten2D, Activation, Dense, Dropout, AlphaDropout, Maxout, Mip, Xip, Combine
from .conv import conv1d_transpose, conv1d_transpose_nn, conv_transpose, \
				  separable_conv, separable_conv_transpose, depthwise_conv, depthwise_conv_transpose
from .conv import Conv1DTranspose, Conv1DTransposeNn, ConvTranspose, \
				  SeparableConv, SeparableConvTranspose, DepthwiseConv, DepthwiseConvTranspose
from .pooling import max_pooling, average_pooling, pooling, \
                     global_pooling, global_average_pooling, global_max_pooling, \
                     fractional_pooling
from .pooling import MaxPooling, AveragePooling, Pooling, \
                     GlobalPooling, GlobalAveragePooling, GlobalMaxPooling, \
                     FractionalPooling
from .roi import roi_pooling_layer, non_max_suppression
from .resize import subpixel_conv, resize_bilinear_additive, resize_nn, resize_bilinear, depth_to_space, \
					SubpixelConv, ResizeBilinearAdditive, ResizeNn, ResizeBilinear, DepthToSpace, \
					IncreaseDim, Reshape, upsample, Upsample, Crop
from .pyramid import pyramid_pooling, aspp
from .drop_block import dropblock
from .drop_block import Dropblock
