"""  Loss as a Metrics to be used in research pipelines added with `run=True` """

import numpy as np

from .base import Metrics


class Loss(Metrics):
    """
    This is a helper class to aggregate losses from pipelines
    that are used in Research objects with `run=True`,
    like test pipelines

    Parameters
    ----------
    loss : float
        loss value obtained from model
    """

    def __init__(self, loss, batch_len):
        super().__init__()

        self.losses = [loss]
        self.batch_lengths = [batch_len]

        def agg_loss(args):
            losses, blens = args
            return np.sum(np.asarray(losses) * np.asarray(blens)) / np.sum(blens)

        self._agg_fn_dict.update(mean=agg_loss)

        def batchwise_loss(args):
            losses, _ = args
            return losses

        self._agg_fn_dict.update(batchwise=batchwise_loss)

    def append(self, metrics):
        """ Extend with data from another metrics. """
        self.losses.extend(metrics.losses)
        self.batch_lengths.extend(metrics.batch_lengths)

    def loss(self):
        return self.losses, self.batch_lengths
