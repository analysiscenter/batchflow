""" Core loss functions """
import torch
import torch.nn as nn


class CrossEntropyLoss(nn.CrossEntropyLoss):
    """ Custom loss which casts target dtype if needed """
    def forward(self, input, target):
        target = target.to(dtype=torch.long)
        return super().forward(input, target)



class Weighted(nn.Module):
    """ Combine multiple losses into one with weights by summation.

    Parameters
    ----------
    losses : sequence
        Instances of loss functions.
    weights : sequence of numbers
        Multiplier for each loss during summation. Default value is 1 over number of losses for each loss.
    starts : sequence of ints
        Iteration to start computation for each of the losses. Default value is 0 for each loss.
    """
    def __init__(self, losses, weights=None, starts=None):
        super().__init__()
        self.losses = losses
        self.n = len(losses)

        if weights is not None:
            assert len(weights) == self.n
            self.weights = weights
        else:
            self.weights = (1 / self.n,) * self.n

        if starts is not None:
            assert len(starts) == self.n
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
