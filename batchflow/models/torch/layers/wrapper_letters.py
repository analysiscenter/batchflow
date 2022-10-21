""" Classes to wrap large (named) blocks as letters to use in layout. """
from torch import nn



class Branch(nn.Module):
    """ Add side branch to a :class:`~..MultiLayer`.
    Used as letters `R`/`B` in layouts.
    Can be ended (and combined with the main flow) with `+`, `*`, `|` operations.
    """
    def __init__(self, inputs=None, **kwargs):
        super().__init__()

        if kwargs.get('layout') or kwargs.get('base_block'):
            from ..blocks import Block
            self.layer = Block(inputs=inputs, **kwargs)
        else:
            self.layer = nn.Identity()

    def forward(self, x):
        return self.layer(x)


class AttentionWrapper(nn.Module):
    """ Attention based on tensor itself.
    Used as `S` letter in layouts.

    Parameters
    ----------
    attention_mode : str or callable
        If callable, then directly applied to the input tensor.
        If str, then one of predefined attention layers:

            If `se`, then squeeze and excitation.
            Hu J. et al. "`Squeeze-and-Excitation Networks <https://arxiv.org/abs/1709.01507>`_"

            If `scse`, then concurrent spatial and channel squeeze and excitation.
            Roy A.G. et al. "`Concurrent Spatial and Channel ‘Squeeze & Excitation’
            in Fully Convolutional Networks <https://arxiv.org/abs/1803.02579>`_"

            If `ssa`, then simple self attention.
            Wang Z. et al. "'Less Memory, Faster Speed: Refining Self-Attention Module for Image
            Reconstruction <https://arxiv.org/abs/1905.08008>'_"

            If `bam`, then bottleneck attention module.
            Jongchan Park. et al. "'BAM: Bottleneck Attention Module
            <https://arxiv.org/abs/1807.06514>'_"

            If `cbam`, then convolutional block attention module.
            Sanghyun Woo. et al. "'CBAM: Convolutional Block Attention Module
            <https://arxiv.org/abs/1807.06521>'_"

            If `fpa`, then feature pyramid attention.
            Hanchao Li, Pengfei Xiong, Jie An, Lingxue Wang.
            Pyramid Attention Network for Semantic Segmentation <https://arxiv.org/abs/1805.10180>'_"

            If `sac`, then split attention.
            Hang Zhang et al. "`ResNeSt: Split-Attention Networks
            <https://arxiv.org/abs/2004.08955>`_"
    """
    def __init__(self, inputs=None, attention='se', **kwargs):
        super().__init__()
        self.attention = attention

        if attention in self.ATTENTIONS:
            op = self.ATTENTIONS[attention]
            self.op = op(inputs, **kwargs)
        elif callable(attention):
            self.op = attention(inputs, **kwargs)
        else:
            available_attentions = list(self.ATTENTIONS.keys())
            raise ValueError(f'Attention must be a callable or one of {available_attentions}, got {attention} instead!')

    def forward(self, inputs):
        return self.op(inputs)

    def extra_repr(self):
        """ Report used attention in a repr. """
        if isinstance(self.attention, (str, bool)):
            return f'op={self.op.__class__.__name__}'
        return f'op=callable {self.attention.__name__}'


    @staticmethod
    def identity(inputs, **kwargs):
        """ Return tensor unchanged. """
        _ = inputs, kwargs
        return nn.Identity()

    @staticmethod
    def squeeze_and_excitation(inputs, ratio=4, **kwargs):
        """ Squeeze and excitation. """
        from ..blocks import SEBlock
        return SEBlock(inputs=inputs, ratio=ratio, **kwargs)

    @staticmethod
    def scse(inputs, ratio=2, **kwargs):
        """ Concurrent spatial and channel squeeze and excitation. """
        from ..blocks import SCSEBlock
        return SCSEBlock(inputs=inputs, ratio=ratio, **kwargs)

    @staticmethod
    def ssa(inputs, ratio=8, **kwargs):
        """ Simple Self Attention. """
        from ..blocks import SimpleSelfAttention
        return SimpleSelfAttention(inputs=inputs, ratio=ratio, **kwargs)

    @staticmethod
    def emha(inputs, ratio=4, num_heads=8, **kwargs):
        """ Efficient Multi Head Attention. """
        from ..blocks import EfficientMultiHeadAttention
        return EfficientMultiHeadAttention(inputs=inputs, ratio=ratio, num_heads=num_heads, **kwargs)

    @staticmethod
    def bam(inputs, ratio=16, **kwargs):
        """ Bottleneck Attention Module. """
        from ..blocks import BAM
        return BAM(inputs=inputs, ratio=ratio, **kwargs)

    @staticmethod
    def cbam(inputs, ratio=16, **kwargs):
        """ Convolutional Block Attention Module. """
        from ..blocks import CBAM
        return CBAM(inputs=inputs, ratio=ratio, **kwargs)

    @staticmethod
    def fpa(inputs, pyramid_kernel_size=(7, 5, 3), bottleneck=False, **kwargs):
        """ Feature Pyramid Attention. """
        from ..blocks import FPA
        return FPA(inputs=inputs, pyramid_kernel_size=pyramid_kernel_size, bottleneck=bottleneck, **kwargs)

    @staticmethod
    def sac(inputs, radix=2, cardinality=1, **kwargs):
        """ Split-Attention Block. """
        from ..blocks import SplitAttentionConv
        return SplitAttentionConv(inputs=inputs, radix=radix, cardinality=cardinality, **kwargs)

    ATTENTIONS = {
        squeeze_and_excitation: ['se', 'squeeze_and_excitation', 'SE', True],
        scse: ['scse', 'SCSE'],
        ssa: ['ssa', 'SSA'],
        emha: ['emha', 'EMHA', 'transformer'],
        bam: ['bam', 'BAM'],
        cbam: ['cbam', 'CBAM'],
        fpa: ['fpa', 'FPA'],
        identity: ['identity', None, False],
        sac: ['sac', 'SAC']
    }
    ATTENTIONS = {alias: getattr(method, '__func__') for method, aliases in ATTENTIONS.items() for alias in aliases}
