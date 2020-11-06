""" Implementation of the Lovasz Softmax loss.
Maxim Berman, et al "`The Lovász-Softmax loss: A tractable surrogate for the optimization
of the intersection-over-union measure in neural networks <https://arxiv.org/abs/1705.08790>`_"

Heavily based on author's implementation: https://github.com/bermanmaxim/LovaszSoftmax
"""
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable


class BinaryLovaszLoss(nn.Module):
    """ Compute binary Lovasz loss.

    Parameters
    ----------
    per_image : bool
        Whether to aggregate loss at individual items or at entire batch.
    ignore : None or int
        Class to exclude from computations
    """
    def __init__(self, per_image=False, ignore=None):
        super().__init__()
        self.per_image = per_image
        self.ignore = ignore

    def forward(self, prediction, target):
        if self.per_image:
            lst = [self.compute_loss(*self.flatten(logit.unsqueeze(0), label.unsqueeze(0)))
                   for logit, label in zip(prediction, target)]
            loss = torch.mean(torch.stack(lst), dim=0)

        else:
            loss = self.compute_loss(*self.flatten(prediction, target))
        return loss

    def flatten(self, scores, labels):
        """ Flatten predictions and true labels and remove ignored class. """
        scores = scores.view(-1)
        labels = labels.view(-1)
        if self.ignore is None:
            return scores, labels

        mask = labels != self.ignore
        return scores[mask], labels[mask]

    def compute_loss(self, logits, labels):
        """ Takes in flattened binary tensors and outputs binary Lovasz loss. """
        if len(labels) == 0:
            # only void pixels, the gradients should be 0
            return logits.sum() * 0.0

        signs = 2.0 * labels.float() - 1.0
        errors = 1.0 - logits * Variable(signs)
        errors_sorted, perm = torch.sort(errors, dim=0, descending=True)

        gt_sorted = labels[perm.data]
        grad = lovasz_grad(gt_sorted)
        loss = torch.dot(F.relu(errors_sorted), Variable(grad))
        return loss



class LovaszLoss(nn.Module):
    """ Compute Lovasz Softmax loss.

    Parameters
    ----------
    per_image : bool
        Whether to aggregate loss at individual items or at entire batch.
    ignore : None or int
        Class to exclude from computations
    ignore_missing_classes : bool
        Whether to include missing in computations classes for averaging purposes.
    """
    def __init__(self, per_image=False, ignore=None, ignore_missing_classes=True):
        super().__init__()
        self.per_image = per_image
        self.ignore = ignore
        self.ignore_missing_classes = ignore_missing_classes

    def forward(self, prediction, target):
        if self.per_image:
            lst = [self.compute_loss(*self.flatten(logit.unsqueeze(0), label.unsqueeze(0)))
                   for logit, label in zip(prediction, target)]
            loss = torch.mean(torch.stack(lst), dim=0)

        else:
            loss = self.compute_loss(*self.flatten(prediction, target))
        return loss

    def flatten(self, probas, labels):
        """ Flatten predictions and true labels and remove ignored class. """
        # Assume output of a sigmoid layer
        if probas.dim() == 3:
            probas = probas.unsqueeze(1)

        # Flatten
        C = probas.size(1)
        probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C
        labels = labels.view(-1) # => B * H * W

        # Optional filtration
        if self.ignore is None:
            return probas, labels

        mask = labels != self.ignore
        return probas[mask], labels[mask]

    def compute_loss(self, probas, labels):
        """ Takes in flattened tensors and outputs binary Lovasz loss. """
        if probas.numel() == 0:
            # only void pixels, the gradients should be 0
            return probas * 0.0

        C = probas.size(1)

        per_class_losses = []
        for c in range(C):
            gt_at_class = (labels == c).float()  # foreground for class c

            # Don't count missing (in true labels) classes
            if self.ignore_missing_classes and gt_at_class.sum() == 0:
                continue

            class_pred = probas[:, c]

            errors = (Variable(gt_at_class) - class_pred).abs()
            errors_sorted, perm = torch.sort(errors, dim=0, descending=True)

            class_loss = torch.dot(errors_sorted, Variable(lovasz_grad(gt_at_class[perm.data])))
            per_class_losses.append(class_loss)
        return torch.mean(torch.stack(per_class_losses), dim=0)


def lovasz_grad(gt_sorted):
    """ Compute gradient of the Lovasz extension w.r.t sorted errors. See Alg. 1 in paper. """
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union

    p = len(gt_sorted)
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard
