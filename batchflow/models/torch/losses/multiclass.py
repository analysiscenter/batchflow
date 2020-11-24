""" Losses that work with multiclass tasks. """
import torch
import torch.nn as nn
import torch.nn.functional as F



class Dice(nn.Module):
    """ Sørensen-Dice Coefficient as a loss function. Sudre C. et al. "`Generalised Dice overlap as a deep
    learning loss function for highly unbalanced segmentations <https://arxiv.org/abs/1707.03237>`_".

    Predictions are passed through a softmax function to obtain probabilities.
    """
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, prediction, target):
        num_classes = prediction.shape[1]
        ndims = target.ndimension()

        prediction = F.softmax(prediction, dim=1)

        target = target.long()
        target = torch.eye(num_classes)[target.squeeze(1)]
        target = target.permute(0, -1, *tuple(range(1, ndims - 1))).float()
        target = target.to(prediction.device).type(prediction.type())

        dims = (0,) + tuple(range(2, ndims))
        intersection = torch.sum(prediction * target, dims)
        cardinality = torch.sum(prediction + target, dims)
        dice_coeff = (2. * intersection / (cardinality + self.eps)).mean()
        return 1 - dice_coeff
