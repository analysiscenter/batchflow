""" Layer implementing Pixel Shuffle for an arbitrary dimensionality.
Shi et al. "`Real-Time Single Image and Video Super-Resolution
Using an Efficient Sub-Pixel Convolutional Neural Network_
<https://arxiv.org/pdf/1609.05158.pdf>`"
"""
from torch import nn



class PixelShuffle(nn.Module):
    """ Rearranges elements in a tensor of shape [B, C * r^N, D1, D2, ... DN]
    to a tensor of shape [B, C, D1 * r, D2 * r, ... DN * r]
    where r is an upscale factor.

    Parameters
    ----------
    upscale_factor : int
        factor to increase spatial resolution by.
    """
    def __init__(self, upscale_factor):
        super().__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x):
        batch_size, channels, *dims = x.size()                                         # (B, C * r^N, H, W)
        ndims = len(dims)
        channels //= (self.upscale_factor**ndims)

        factor_expand = [self.upscale_factor] * ndims
        x = x.contiguous().view(batch_size, channels,                                  # (B, C, r, r, H, W)
                                *factor_expand, *dims)

        permute = [None] * (2 * ndims)
        permute[::2] = range(ndims + 2, 2 * ndims + 2)
        permute[1::2] = range(2, ndims + 2)
        x = x.permute(0, 1, *permute).contiguous()                                     # (B, C, H, r, W, r)

        out_dims = [dim * self.upscale_factor for dim in dims]
        x = x.view(batch_size, channels, *out_dims).contiguous()                       # (B, C, H * r, W * r)
        return x

    def extra_repr(self):
        return f'upscale_factor={self.upscale_factor}'



class PixelUnshuffle(nn.Module):
    """ Rearranges elements in a tensor of shape [B, C, D1 * r, D2 * r, ... DN * r]
    to a tensor of shape [B, C * r^N, D1, DN], where r is a downscale factor.

    Parameters
    ----------
    downscale_factor : int
        factor to decrease spatial resolution by.
    """
    def __init__(self, downscale_factor):
        super().__init__()
        self.downscale_factor = downscale_factor

    def forward(self, x):
        x = x.contiguous()                                                             # (B, C, H * r, B * r)
        batch_size, channels, *dims = x.size()
        ndims = len(dims)

        out_dims = [dim // self.downscale_factor for dim in dims]
        reshape = [None] * (2 * ndims)
        reshape[::2] = out_dims
        reshape[1::2] = [self.downscale_factor for i in range(ndims)]

        x = x.view(batch_size, channels, *reshape)                                     # (B, C, H, r, W, r)

        permute = [None] * (2 * ndims)
        permute[:2 * ndims] = range(3, 2 * ndims + 2, 2)
        permute[2 * ndims:] = range(2, 2 * ndims + 1, 2)
        x = x.permute(0, 1, *permute).contiguous()                                     # (B, C, r, r, H, W)

        out_channels = channels * self.downscale_factor ** ndims
        x = x.view(batch_size, out_channels, *out_dims).contiguous()                   # (B, C * r^N, H, W)
        return x

    def extra_repr(self):
        return f'downscale_factor={self.downscale_factor}'
