""" Transformer blocks from various NN architectures. """
from .core import Block
from ..utils import get_num_channels



class SegFormerBlock(Block):
    """ SegFormer block: semantic segmentation block, based on transformer architectures.
    Essentially, a sequence of efficient self attention and MLP block.
    Enze Xie et al. "`SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    <https://arxiv.org/abs/2105.15203>`_"

    Key difference to other transformer-based networks is the absence of positional encoding.
    Pairing this block, authors propose to use simple MLP decoder: it is available as :class:`~.MLPDecoderModule`.

    Parameters
    ----------
    ratio : int
        Spatial reduction ratio for self attention.
    num_heads : int
        Number of parallel attention heads. Must be a divisor of input number of channels.
    mlp_expansion : int
        Expansion ratio for channels in MLP block.
    """
    def __init__(self, inputs=None, layout='RnS! Rnccac!', ratio=4, num_heads=8,
                 mlp_expansion=4, drop_path=0.0, layer_scale=1, **kwargs):
        in_channels = get_num_channels(inputs)

        kwargs = {
            'attention': 'emha',
            'self_attention': {'ratio': ratio, 'num_heads': num_heads},

            'channels': [in_channels, in_channels*mlp_expansion, in_channels],
            'kernel_size': [1, 3, 1],
            'groups': [1, in_channels, 1],
            'stride': 1,
            'bias': True,

            'activation': 'GELU',
            'branch_end': {'drop_path': drop_path, 'layer_scale': layer_scale},
            **kwargs
        }
        super().__init__(inputs=inputs, layout=layout, **kwargs)

class MOATBlock(Block):
    """ MOAT block: combination of Mobile Conv blocks (inverted residuals) and transformer-like self attention.
    Yang et al. "`MOAT: Alternating Mobile Convolution and Attention Brings Strong Vision Models
    <https://arxiv.org/abs/2210.01820>`_"

    Parameters
    ----------
    ratio : int
        Spatial reduction ratio for self attention.
    num_heads : int
        Number of parallel attention heads. Must be a divisor of input number of channels.
    expansion : int
        Expansion ratio for channels in MBConv part block.
    """
    def __init__(self, inputs=None, layout='Rnc nac nac! RnS!', ratio=4, num_heads=8,
                 expansion=4, drop_path=0.0, layer_scale=1, **kwargs):
        in_channels = get_num_channels(inputs)

        kwargs = {
            'channels': [in_channels, in_channels*expansion, in_channels],
            'kernel_size': [1, 3, 1],
            'groups': [1, in_channels, 1],
            'stride': 1,
            'bias': False,

            'attention': 'emha',
            'self_attention': {'ratio': ratio, 'num_heads': num_heads},

            'activation': 'GELU',
            'branch_end': {'drop_path': drop_path, 'layer_scale': layer_scale},
            **kwargs
        }
        super().__init__(inputs=inputs, layout=layout, **kwargs)
