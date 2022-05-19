""" Blocks: large parts that implement idea/named entity from popular articles. """
from .core import Block, DefaultBlock, Downsample, Upsample
from .named_blocks import ResBlock, ResNeStBlock, DenseBlock, MBConvBlock, InvResBlock, ConvNeXtBlock
from .attention import SEBlock, SCSEBlock, SimpleSelfAttention, BAM, CBAM, FPA, SelectiveKernelConv, SplitAttentionConv
from .pyramid import PyramidPooling, ASPP, KSAC
from ..layers import Combine # convenience
