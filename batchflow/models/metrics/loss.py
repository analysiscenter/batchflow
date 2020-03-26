"""  Loss as a Metrics to be used in research pipelines added with `run=True` """

import numpy as np

from . import Metrics


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

    def __init__(self, loss):
        super().__init__()

        self.__losses = [loss]
        self._agg_fn_dict.update(mean=np.mean)

    def append(self, metrics):
        """ Extend with data from another metrics"""
        self.__losses.extend(metrics.loss())

    def loss(self):
        return self.__losses[:]
