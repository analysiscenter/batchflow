""" Callbacks to perform on loss plateau. """
import numpy as np

from .base import BaseCallback
from ....exceptions import StopPipeline



class PlateauCallback(BaseCallback):
    """ Detect whether the model loss has plateaued, and perform action if it is.
    Method :meth:`.plateau` is used to check whether the loss has stabilized.
    Method :meth:`.action` is used once the plateau has achieved.
    By default, it simply logs to the `stream` the fact of getting to the plateau.

    Parameters
    ----------
    mode : str
        Mode of computing whether the loss has plateaued or not.
        If `complex`, then we compare means of loss over the last `patience` / 2 iterations and the previous last.
        If `pytorch` or `keras`, we compare the best (minimum) value over the last `patience`
        iterations with the best value overall.
        If `mean`, we compare the mean value over the last `patience` iterations with the best value overall.
    patience : int
        Length of analyzed interval. The bigger, the more stabilized loss should be to considered to be on plateau.
    min_delta : float
        If the difference between compared values (see `mode`) is smaller than this delta, then we consider
        loss to be on plateau. The lower, the more stabilized loss should be considered to be on plateau.
    cooldown : int
        Number of iterations to skip after performing `action`.
    stream : None, callable or str
        If None, then no logging is performed.
        If callable, then used to display message, for example, `print`.
        If str, then must be path to file to write log to.
    """
    def __init__(self, mode='complex', patience=10, min_delta=0.001, cooldown=None, stream=None):
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.cooldown = cooldown or np.Inf

        self.best = np.Inf
        self.cooldown_counter = patience # this way, we ignore the first iterations
        super().__init__(stream=stream)

    def plateau(self):
        """ Check if the loss has plateaued. """
        # On a cooldown from previous update
        if self.cooldown_counter != 0:
            self.cooldown_counter -= 1
            return False

        loss_list = self.model.loss_list
        if self.mode in ['complex']:
            prev = loss_list[-self.patience: -self.patience//2]
            curr = loss_list[-self.patience//2:]
            prev, curr = np.mean(prev), np.mean(curr)

        elif self.mode in ['pytorch', 'keras']:
            curr = np.min(loss_list[-self.patience:])
            prev = self.best

        elif self.mode in ['mean']:
            curr = np.mean(loss_list[-self.patience:])
            prev = self.best

        if curr < (prev - self.min_delta):
            self.best = curr
            return False
        return True

    def on_iter_end(self, **kwargs):
        """ Check if the model loss on plateau, and perform action, if needed.
        Called at the end of :meth:`TorchModel.train`.
        """
        _ = kwargs

        if self.plateau():
            self.action()
            self.cooldown_counter = self.cooldown - 1
            self.stream(f'{self.__class__.__name__} at iteration {self.model.iteration}')

    def action(self):
        """ Action to be performed on plateau. Must be implemented in subclasses. """



class ReduceLROnPlateau(PlateauCallback):
    """ Reduce learning rate by a multiplicative factor, if loss has plateaued.

    Parameters
    ----------
    factor : float
        Value to multiply learning rate on.
    min_lr : float
        Minimum value of the set learning rate.
    args : dict
        The same args as in :class:`PlateauCallback`.
    """
    def __init__(self, mode='complex', patience=10, min_delta=0.001, cooldown=None,
                 factor=0.1, min_lr=0.0, stream=None):
        self.factor = factor
        self.min_lr = min_lr

        super().__init__(mode=mode, patience=patience, min_delta=min_delta, cooldown=cooldown, stream=stream)


    def action(self):
        """ Update the learning rate. """
        for group in self.model.optimizer.param_groups:
            group['lr'] = max(self.min_lr, self.factor * group['lr'])



class EarlyStopping(PlateauCallback):
    """ Stop the model training, if loss has plateaued.

    Parameters
    ----------
    args : dict
        The same args as in :class:`PlateauCallback`.
    """
    def __init__(self, mode='complex', patience=10, min_delta=0.001, cooldown=None, stream=None):
        cooldown = cooldown or patience

        super().__init__(mode=mode, patience=patience, min_delta=min_delta, cooldown=cooldown, stream=stream)

    def action(self):
        """ Raise a signal to stop the pipeline. """
        raise StopPipeline
