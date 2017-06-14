""" Contains the base batch class """

class BaseBatch:
    """ Basr class for batches
    Required to solve circulal module dependencies
    """
    def __init__(self, index):
        self.index = index
        self._data = None
