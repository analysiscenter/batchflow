""" Hamburger layer.
Zhengyang Geng et al. "`Is attention better than matrix decomposition? <https://arxiv.org/abs/2109.04553>`_"
"""
import torch
from torch import nn
import torch.nn.functional as F



class Hamburger(nn.Module):
    """ Hamburger layer: attention by matrix decomposition.
    Zhengyang Geng et al. "`Is attention better than matrix decomposition? <https://arxiv.org/abs/2109.04553>`_"

    The main idea is to use classic matrix decomposition algorithms inside the layer during both train and inference.
    Here we implement only the Non-Negative Matrix Factorization algorithm.

    Code is adopted from authors original implementation
    <https://github.com/Visual-Attention-Network/SegNeXt/blob/main/mmseg/models/decode_heads/ham_head.py>

    Parameters
    ----------
    S : int
        Reduction ratio for the number of channels in inputs tensor.
    R : int
        Size of the hidden dimension for matrix factorization.
    n_train_steps, n_eval_steps : int
        Number of iterative MD steps to perform during train/inference.
    inv_t : number
        Scaling factor for initial guess of matrix decomposition.
    spatial : bool
        Whether to perform attention along spatial or channel axis.
    rand_init : bool
        Whether to init matrix decomposition from scratches at each iteration.
    """
    #pylint: disable=invalid-name
    def __init__(self, inputs=None, S=1, R=64, n_train_steps=6, n_eval_steps=7, inv_t=1, spatial=True, rand_init=True):
        super().__init__()

        self.S, self.R = S, R
        self.n_train_steps, self.n_eval_steps = n_train_steps, n_eval_steps
        self.inv_t = inv_t

        self.device = inputs.device if inputs is not None else 'cpu'
        self.spatial = spatial
        self.rand_init = rand_init


    def forward(self, x):
        B, C, H, W = x.shape

        # (B, C, H, W) -> (B * S, D, N)
        if self.spatial:
            D = C // self.S
            N = H * W
            x = x.view(B * self.S, D, N)
        else:
            D = H * W
            N = C // self.S
            x = x.view(B * self.S, N, D).transpose(1, 2)

        # (S, D, R) -> (B * S, D, R)
        if self.rand_init:
            bases = self.build_bases(B, self.S, D, self.R)
        else:
            if not hasattr(self, 'bases'):
                bases = self.build_bases(1, self.S, D, self.R)
                self.register_buffer('bases', bases)
            bases = self.bases.repeat(B, 1, 1)

        bases, coeff = self.local_inference(x, bases)

        # (B * S, N, R)
        coeff = self.compute_coeff(x, bases, coeff)

        # (B * S, D, R) @ (B * S, N, R)^T -> (B * S, D, N)
        x = torch.bmm(bases, coeff.transpose(1, 2))

        # (B * S, D, N) -> (B, C, H, W)
        if self.spatial:
            x = x.view(B, C, H, W)
        else:
            x = x.transpose(1, 2).view(B, C, H, W)
        return x

    def build_bases(self, B, S, D, R):
        """ Make an initial guess for matrix factorization. """
        bases = torch.rand((B * S, D, R), device=self.device)
        bases = F.normalize(bases, dim=1)
        return bases

    @torch.no_grad()
    def local_inference(self, x, bases):
        """ Multiple updates of `bases` and `coeff` to better match `x`. """
        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        coeff = torch.bmm(x.transpose(1, 2), bases)
        coeff = F.softmax(self.inv_t * coeff, dim=-1)

        steps = self.n_train_steps if self.training else self.n_eval_steps
        for _ in range(steps):
            bases, coeff = self.local_step(x, bases, coeff)
        return bases, coeff

    @torch.no_grad()
    def local_step(self, x, bases, coeff):
        """ Update `bases` and `coeff` to better match `x`. """
        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        numerator = torch.bmm(x.transpose(1, 2), bases)
        # (B * S, N, R) @ [(B * S, D, R)^T @ (B * S, D, R)] -> (B * S, N, R)
        denominator = coeff.bmm(bases.transpose(1, 2).bmm(bases))
        # Multiplicative Update
        coeff = coeff * numerator / (denominator + 1e-6)

        # (B * S, D, N) @ (B * S, N, R) -> (B * S, D, R)
        numerator = torch.bmm(x, coeff)
        # (B * S, D, R) @ [(B * S, N, R)^T @ (B * S, N, R)] -> (B * S, D, R)
        denominator = bases.bmm(coeff.transpose(1, 2).bmm(coeff))
        # Multiplicative Update
        bases = bases * numerator / (denominator + 1e-6)

        return bases, coeff

    def compute_coeff(self, x, bases, coeff):
        """ Update `coeff` to better match `x` with given `bases`. """
        # (B * S, D, N)^T @ (B * S, D, R) -> (B * S, N, R)
        numerator = torch.bmm(x.transpose(1, 2), bases)
        # (B * S, N, R) @ (B * S, D, R)^T @ (B * S, D, R) -> (B * S, N, R)
        denominator = coeff.bmm(bases.transpose(1, 2).bmm(bases))
        # multiplication update
        coeff = coeff * numerator / (denominator + 1e-6)
        return coeff
