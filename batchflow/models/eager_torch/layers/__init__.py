""" PyTorch custom layers. """
# pylint: disable=wildcard-import
from .conv_block import BaseConvBlock, ConvBlock, update_layers
from .core import Flatten, Dense, Activation, Dropout, AlphaDropout, BatchNorm
from .conv import Conv, ConvTranspose, DepthwiseConv, DepthwiseConvTranspose, SeparableConv, SeparableConvTranspose
from .pooling import MaxPool, AvgPool, Pool, AdaptiveMaxPool, AdaptiveAvgPool, AdaptivePool, \
					 GlobalPool, GlobalMaxPool, GlobalAvgPool
from .resize import IncreaseDim, ReduceDim, Reshape, Interpolate, PixelShuffle, SubPixelConv, \
					SideBlock, Upsample, Combine, SEBlock, Crop
from .pyramid import PyramidPooling, ASPP