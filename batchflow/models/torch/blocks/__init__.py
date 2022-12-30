""" Blocks: large parts that implement idea/named entity from popular articles. """
from .core import Block, DefaultBlock, Downsample, Upsample
from .named_blocks import (VGGBlock, ResBlock, BottleneckBlock, ResNeStBlock, DenseBlock, MBConvBlock, InvResBlock,
                           ConvNeXtBlock, MSCANBlock, InternImageBlock)
from .transformer_blocks import SegFormerBlock, MOATBlock
from .attention import (SEBlock, SCSEBlock, SimpleSelfAttention, EfficientMultiHeadAttention,
                        BAM, CBAM, FPA, SelectiveKernelConv, SplitAttentionConv)
from .pyramid import PyramidPooling, ASPP, KSAC
from ..layers import Combine, Branch # convenience
