""" Losses for binary predictions. """
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F



class BCE(nn.Module):
    """ Binary cross-entropy that allows int/float for `pos_weight`, unlike native PyTorch implementation.

    Parameters
    ----------
    pos_weight : number
        A weight for positive examples.
    """
    def __init__(self, pos_weight=1, reduction='mean'):
        super().__init__()
        self.pos_weight = pos_weight
        self.reduction = reduction

    def forward(self, prediction, target):
        pos_weight = self.pos_weight * torch.ones(1, device=prediction.device)
        loss = F.binary_cross_entropy_with_logits(prediction, target, pos_weight=pos_weight,
                                                  reduction=self.reduction)
        return loss


class TopK(nn.Module):
    """ Binary cross entropy computed only over k% worst elements.

    Parameters
    ----------
    pos_weight : number
        A weight for positive examples.
    k : number
        Percent of worst examples to include in loss computation.
    """
    def __init__(self, pos_weight=1, k=10):
        super().__init__()
        self.pos_weight = pos_weight
        self.k = k

    def forward(self, prediction, target):
        pos_weight = self.pos_weight * torch.ones(1, device=prediction.device)
        loss = F.binary_cross_entropy_with_logits(prediction, target,
                                                  pos_weight=pos_weight,
                                                  reduction='none')
        n = np.prod(loss.shape)
        loss, _ = torch.topk(loss.view(-1), int(n * self.k) // 100, sorted=False)
        return loss.mean()



class Dice(nn.Module):
    """ Sørensen-Dice Coefficient as a loss function. Sudre C. et al. "`Generalised Dice overlap as a deep
    learning loss function for highly unbalanced segmentations <https://arxiv.org/abs/1707.03237>`_".

    Predictions are passed through a sigmoid function to obtain probabilities.
    """
    def __init__(self, eps=1e-7, apply_sigmoid=True):
        super().__init__()
        self.eps = eps
        self.apply_sigmoid = apply_sigmoid

    def forward(self, prediction, target):
        if self.apply_sigmoid:
            prediction = torch.sigmoid(prediction)
        dice_coeff = 2. * (prediction * target).sum() / (prediction.sum() + target.sum() + self.eps)
        return 1 - dice_coeff


class PenaltyDice(Dice):
    """ Modification of a Dice loss with additional weight towards false positives and false negatives.
    Yang Su et al. "` Major Vessel Segmentation on X-ray Coronary Angiography using Deep Networks with a
    Novel Penalty Loss Function <https://openreview.net/forum?id=H1lTh8unKN>`_".

    Parameters
    ----------
    k : number
        Penalty coefficient: the bigger, the more weight is put on false positives and negatives.
    """
    def __init__(self, k=3):
        super().__init__()
        self.k = k

    def forward(self, prediction, target):
        dice_loss = super().forward(prediction, target)
        return dice_loss / (1 + self.k*(1 - dice_loss))


class LogDice(Dice):
    """ Modification of a Dice loss with additional emphasis on smaller objects.
    Wong K. et al. "` 3D Segmentation with Exponential Logarithmic Loss for Highly Unbalanced
    Object Sizes <https://arxiv.org/abs/1809.00076>`_".

    Parameters
    ----------
    gamma : number
        Modulating factor. The bigger, the more weight is put on less accurate predictions.
    """
    def __init__(self, gamma=2.):
        super().__init__()
        self.gamma = gamma

    def forward(self, prediction, target):
        dice_loss = super().forward(prediction, target)
        dice_coeff = 1 - dice_loss
        return torch.pow(-torch.log(dice_coeff), self.gamma)



class Tversky(nn.Module):
    """ Generalization of Dice loss with finer control over false positives and negatives weights.
    Salehi S. et al. "` Tversky loss function for image segmentation using 3D fully convolutional
    deep networks <https://arxiv.org/abs/1706.05721>`_".

    If `alpha` and `beta` equal 0.5, identical to Dice.
    If `alpha` and `beta` equal 1, identical to Jaccard loss.
    If `alpha` > `beta`, put more weight on false positives, and vice versa.

    Parameters
    ----------
    alpha : number
        Weight for false positive examples.
    beta : number
        Weight for false negative examples.
    """
    def __init__(self, alpha=1., beta=1., eps=1e-7, apply_sigmoid=True):
        super().__init__()
        self.alpha, self.beta = alpha, beta
        self.eps = eps
        self.apply_sigmoid = apply_sigmoid

    def forward(self, prediction, target):
        if self.apply_sigmoid:
            prediction = torch.sigmoid(prediction)

        intersection = (prediction * target).sum()
        false_positive = (prediction * (1 - target)).sum()
        false_negative = ((1 - prediction) * target).sum()
        tversky_coeff = intersection / (intersection + self.alpha * false_positive
                                        + self.beta * false_negative + self.eps)
        return 1 - tversky_coeff


class FocalTversky(Tversky):
    """ Modification of Tversky loss with additional emphasis on harder examples.
    Abraham N. et al. "` A Novel Focal Tversky loss function with improved Attention U-Net for
    lesion segmentation <https://arxiv.org/abs/1810.07842>`_".

    Parameters
    ----------
    gamma : number
        Modulating factor. The bigger, the more weight is put on harder examples.
    """
    def __init__(self, alpha=1., beta=1., gamma=1.3):
        super().__init__(alpha=alpha, beta=beta)
        self.gamma = gamma

    def forward(self, prediction, target):
        tversky_loss = super().forward(prediction, target)
        return torch.pow(tversky_loss, self.gamma)



class SSLoss(nn.Module):
    """ Sensitivity-specificity loss.
    Sensitivity is error over trues, specificity is error over falses: essentially, they correspond to
    `how good are we doing where we should predict True` and `how bad are we doing where we should predict False`.

    Parameters
    ----------
    r : number
        Weight for specificity; weight for sensitivity is 1 - `r`.
    """
    def __init__(self, r=0.1, eps=1e-7, apply_sigmoid=True):
        super().__init__()
        self.r = r
        self.eps = eps
        self.apply_sigmoid = apply_sigmoid

    def forward(self, prediction, target):
        if self.apply_sigmoid:
            prediction = torch.sigmoid(prediction)
        inverse = 1 - target

        squared_error = (target - prediction)**2
        specificity = (squared_error * target).sum() / (target.sum() + self.eps)
        sensitivity = (squared_error * inverse).sum() / (inverse.sum() + self.eps)

        return self.r * specificity + (1 - self.r) * sensitivity
