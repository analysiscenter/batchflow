""" Callbacks to perform on loss plateau. """
import numpy as np

from .base import BaseCallback



class PlateauCallback(BaseCallback):
    """ !!. """
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
        """ Perform the callback action, if on plateau. """
        _ = kwargs

        if self.plateau():
            self.action()
            self.cooldown_counter = self.cooldown - 1
            self.stream(f'{self.__class__.__name__} at iteration {self.model.iteration}')

    def action(self):
        """ Must be implemented in subclasses. """



class ReduceLROnPlateau(PlateauCallback):
    """ Reduce learning rate by a multiplicative factor, if loss has plateaued. """
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
    """ Stop the model training, if loss has plateaued. """
    def __init__(self, mode='complex', patience=10, min_delta=0.001, cooldown=None, stream=None):
        cooldown = cooldown or patience

        super().__init__(mode=mode, patience=patience, min_delta=min_delta, cooldown=cooldown, stream=stream)

    def action(self):
        """ Raise a signal to stop the pipeline. """
        raise StopIteration
