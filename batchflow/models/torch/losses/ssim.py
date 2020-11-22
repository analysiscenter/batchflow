""" Structural similarity as loss function.
"Image Quality Assessment: From Error Visibility to Structural Similarity" IEEE
Transactions on Image Processing Vol. 13. No. 4. April 2004.
"""
from math import exp

import torch
import torch.nn as nn
import torch.nn.functional as F


class SSIM(nn.Module):
    """ Structural similarity between two images.
    Note that images must be grayscale.

    Heavily inspired by the author's implementation:
    https://ece.uwaterloo.ca/~z70wang/research/ssim/ssim_index.m

    Parameters
    ----------
    kernel_size : int
        Size of filtering window.
    sigma : number
        Variance of Gaussian kernel.
    """
    def __init__(self, kernel_size=11, sigma=1.5):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma

        self.kernel = None

    def forward(self, prediction, target):
        if self.kernel is None:
            self.kernel = self.create_kernel(prediction.size()[1])
            self.kernel = self.kernel.type_as(prediction).to(prediction.device)

        return 1 - self.compute_ssim(prediction, target, return_sigma=False).mean()


    def create_kernel(self, channel):
        """ Create Gaussian kernel. """
        kernel_1d = torch.Tensor([exp(-(x - self.kernel_size // 2)**2 / (2*self.sigma**2))
                                  for x in range(self.kernel_size)])
        kernel_1d = (kernel_1d / kernel_1d.sum()).unsqueeze(1)

        kernel_2d = kernel_1d.mm(kernel_1d.t()).float().unsqueeze(0).unsqueeze(0)
        kernel = kernel_2d.expand(channel, 1, self.kernel_size, self.kernel_size).contiguous()
        return kernel

    def compute_ssim(self, prediction, target, return_sigma=False):
        """ Compute structural similarity map. """
        channel = prediction.size()[1]

        mu1 = F.conv2d(prediction, self.kernel, padding=self.kernel_size // 2, groups=channel)
        mu2 = F.conv2d(target, self.kernel, padding=self.kernel_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1*mu2

        sigma1_sq = F.conv2d(prediction*prediction, self.kernel, padding=self.kernel_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(target*target, self.kernel, padding=self.kernel_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(prediction*target, self.kernel, padding=self.kernel_size // 2, groups=channel) - mu1_mu2

        mu_map = (2*mu1_mu2 + 0.0001) / (mu1_sq + mu2_sq + 0.0001)
        sigma_map = (2*sigma12 + 0.0009) / (sigma1_sq + sigma2_sq + 0.0009)
        ssim_map = mu_map * sigma_map

        if return_sigma:
            return ssim_map, sigma_map
        return ssim_map


class MSSIM(SSIM):
    """ Compute SSIM on multiple scales and return the weighted average.
    Note that images must be grayscale.

    Heavily inspired by the author's implementation:
    http://www.cns.nyu.edu/~lcv/ssim/msssim.zip
    """
    WEIGHTS = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333] # from the author's code

    def __init__(self, kernel_size=11, sigma=1.5):
        super().__init__()

        self.pooling = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, prediction, target):
        if self.kernel is None:
            self.kernel = self.create_kernel(prediction.size()[1])
            self.kernel = self.kernel.type_as(prediction).to(prediction.device)

        ssims, sigmas = [], []
        for _ in self.WEIGHTS:
            ssim_map, sigma_map = self.compute_ssim(prediction, target, return_sigma=True)
            ssims.append(ssim_map.mean())
            sigmas.append(sigma_map.mean())

            prediction = self.pooling(prediction)
            target = self.pooling(target)

        output = 1
        for sigma, weight in zip(sigmas[:4], self.WEIGHTS[:4]):
            output *= sigma ** weight
        output *= ssims[-1] ** self.WEIGHTS[-1]
        return 1 - output
