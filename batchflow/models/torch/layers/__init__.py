""" PyTorch custom layers. """
# pylint: disable=wildcard-import
from .conv_block import BaseConvBlock, ConvBlock, update_layers
from .core import Flatten, Dense, Dropout, AlphaDropout, BatchNorm
from .activation import Activation
from .conv import Conv, ConvTranspose, DepthwiseConv, DepthwiseConvTranspose, SeparableConv, SeparableConvTranspose
from .pooling import MaxPool, AvgPool, Pool, AdaptiveMaxPool, AdaptiveAvgPool, AdaptivePool, \
					 GlobalPool, GlobalMaxPool, GlobalAvgPool, ChannelPool
from .resize import IncreaseDim, Reshape, Interpolate, PixelShuffle, SubPixelConv, \
					Upsample, Combine, Crop
from .attention import SelfAttention, SEBlock, SCSEBlock, SimpleSelfAttention, BAM, CBAM, FPA, SelectiveKernelConv
from .modules import PyramidPooling, ASPP, KSAC
