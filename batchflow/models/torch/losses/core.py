""" Core loss functions """
import torch
import torch.nn as nn


class CrossEntropyLoss(nn.Module):
    """ Custom loss which casts target dtype if needed.
    Additionally, allows string specifiers for `weight` parameter.

    Parameters
    ----------
    squeeze : bool
        Whether to remove the channel axis of targets.
    weight : str, callable or torch.Tensor
        If callable, then used on the support of a given class.
        If one of `dynamic`, `inverse` or `adaptive`, then weight is inversely proportional to the support of a class.
        If `proportional`, then weight is the same as the support of a class.
        If Tensor, then uses the same semantics as :class:`torch.nn.CrossEntropyLoss` implementation.
    other parameters
        The same as :class:`torch.nn.CrossEntropyLoss` arguments.
    """
    def __init__(self, squeeze=False, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean'):
        super().__init__()
        self.kwargs = {
            'weight': weight,
            'size_average': size_average,
            'ignore_index': ignore_index,
            'reduce': reduce,
            'reduction': reduction
        }

        self.squeeze = squeeze
        self.weight = weight
        self.dynamic = isinstance(weight, str) or callable(weight)

    def forward(self, prediction, target):
        # pylint: disable=not-callable
        kwargs = dict(self.kwargs)
        target = target.to(dtype=torch.long)

        if self.squeeze:
            target = target.squeeze(1)

        if self.dynamic:
            num_classes = prediction.shape[1]

            weights = []
            for i in range(num_classes):
                support = (target == i).sum() + 1

                if self.weight in ['dynamic', 'inverse', 'adaptive']:
                    weight = 1 / float(support)
                elif self.weight in ['proportional']:
                    weight = float(support)
                elif callable(self.weight):
                    weight = self.weight(support)
                weights.append(weight)

            weights = torch.tensor(weights, device=prediction.device)
            weights = weights / weights.sum()
            kwargs['weight'] = weights

        loss_func = nn.CrossEntropyLoss(**kwargs)
        return loss_func(prediction, target)



class Weighted(nn.Module):
    """ Combine multiple losses into one with weights by summation.

    Parameters
    ----------
    losses : sequence
        Instances of loss functions.
    weights : sequence of numbers
        Multiplier for each loss during summation. Default value is 1 over the number of losses for each loss.
    starts : sequence of ints
        Iteration to start computation for each of the losses. Default value is 0 for each loss.
    """
    def __init__(self, losses, weights=None, starts=None):
        super().__init__()
        self.losses = losses
        self.n = len(losses)

        if weights is not None:
            if len(weights) != self.n:
                raise ValueError('A weight must be provided for each of the losses!')
            self.weights = weights
        else:
            self.weights = (1 / self.n,) * self.n

        if starts is not None:
            if len(starts) != self.n:
                raise ValueError('Starting iteration must be provided for each of the losses!')
            self.starts = starts
        else:
            self.starts = (0,) * self.n

        self.iter = 0

    def forward(self, prediction, target):
        loss = 0
        for loss_func, weight, start_iter in zip(self.losses, self.weights, self.starts):
            if self.iter >= start_iter:
                loss += weight * loss_func(prediction, target)

        self.iter += 1
        return loss
