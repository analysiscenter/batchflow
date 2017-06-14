""" Contains the base batch class """
from .decorators import action


class BaseBatch:
    """ Basr class for batches
    Required to solve circulal module dependencies
    """
    def __init__(self, index):
        self.index = index
        self._data = None

    @action
    def load(self, src, fmt=None):
        """ Load data from a file or another data source """
        raise NotImplementedError()

    @action
    def dump(self, dst, fmt=None):
        """ Save batch data to disk """
        raise NotImplementedError()

    @action
    def save(self, *args, **kwargs):
        """ Save batch data to a file (an alias for dump method)"""
        return self.dump(*args, **kwargs)
