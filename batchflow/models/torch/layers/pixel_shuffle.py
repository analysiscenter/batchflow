""" Layers implementing Pixel Shuffle and Pixel Unshuffle for an arbitrary dimensionality.
Shi et al. "`Real-Time Single Image and Video Super-Resolution
Using an Efficient Sub-Pixel Convolutional Neural Network_
<https://arxiv.org/pdf/1609.05158.pdf>`"
"""
from torch import nn



class PixelShuffle(nn.Module):
    """ Extends `torch.nn.PixelShuffle` to and arbitrary number of dimensions.
    Rearranges elements in a tensor of shape [B, C * r^N, D1, D2, ... DN]
    to a tensor of shape [B, C, D1 * r, D2 * r, ... DN * r]
    where r is an upscale factor.

    Parameters
    ----------
    upscale_factor : int
        Factor to increase spatial resolution by.
    """
    def __init__(self, upscale_factor):
        super().__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x):
        batch_size, channels, *dims = x.size()                                         # (B, C * r^2, H, W)
        ndims = len(dims)

        if channels % (self.upscale_factor**ndims) != 0:
            raise ValueError("PixelShuffle expects input of shape [B, C * r^N, D1, D2, ... DN]")
        out_channels = channels // self.upscale_factor**ndims

        factor_expand = [self.upscale_factor] * ndims
        x = x.contiguous().view(batch_size, out_channels,                              # (B, C, r, r, H, W)
                                *factor_expand, *dims)

        permute_dims = [None] * (2 * ndims)
        permute_dims[::2] = range(ndims + 2, 2 * ndims + 2)
        permute_dims[1::2] = range(2, ndims + 2)
        x = x.permute(0, 1, *permute_dims).contiguous()                                # (B, C, H, r, W, r)

        out_dims = [dim * self.upscale_factor for dim in dims]
        x = x.view(batch_size, out_channels, *out_dims).contiguous()                   # (B, C, H * r, W * r)
        return x

    def extra_repr(self):
        return f'upscale_factor={self.upscale_factor}'



class PixelUnshuffle(nn.Module):
    """ Extends `torch.nn.PixelUnShuffle` to and arbitrary number of dimensions.
    Rearranges elements in a tensor of shape [B, C, D1 * r, D2 * r, ... DN * r]
    to a tensor of shape [B, C * r^N, D1, DN], where r is a downscale factor.

    Parameters
    ----------
    downscale_factor : int
        Factor to decrease spatial resolution by.
    """
    def __init__(self, downscale_factor):
        super().__init__()
        self.downscale_factor = downscale_factor

    def forward(self, x):
        x = x.contiguous()                                                             # (B, C, H * r, W * r)
        batch_size, channels, *dims = x.size()
        ndims = len(dims)

        for dim in dims:
            if dim % self.downscale_factor != 0:
                raise ValueError("PixelUnshuffle expects input of shape [B, C, D1 * r, D2 * r, ... DN * r]")
        out_dims = [dim // self.downscale_factor for dim in dims]

        shape_dims = [None] * 2 * ndims
        shape_dims[::2] = out_dims
        shape_dims[1::2] = [self.downscale_factor] * ndims

        x = x.view(batch_size, channels, *shape_dims)                                  # (B, C, H, r, W, r)

        permute_dims = [None] * 2 * ndims
        permute_dims[:2 * ndims] = range(3, 2 * ndims + 2, 2)
        permute_dims[2 * ndims:] = range(2, 2 * ndims + 1, 2)
        x = x.permute(0, 1, *permute_dims).contiguous()                                # (B, C, r, r, H, W)

        out_channels = channels * self.downscale_factor ** ndims
        x = x.view(batch_size, out_channels, *out_dims).contiguous()                   # (B, C * r^N, H, W)
        return x

    def extra_repr(self):
        return f'downscale_factor={self.downscale_factor}'
