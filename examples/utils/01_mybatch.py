"""File contains main class named myBatch and function to generate data."""
import sys

sys.path.append('../..')
from dataset.dataset import action, Batch

class MyBatch(Batch):
    """ Class for load data into regression models

    Attributes:
    ----------
    input_data: numpy array
    Data for training model

    labels: numpy array
    Answers
    """

    def __init__(self, index, *args, **kwargs):
        _ = args, kwargs
        super().__init__(index, *args, **kwargs)
        self.input = None
        self.labels = None

    @property
    def components(self):
        """ Define components """
        return 'input', 'labels'

    @action
    def load(self, src, fmt='blosc', components=None, *args, **kwargs):
        """ Loading data to self.x and self.y

        Parameters:
        ----------
        src: tuple or list
        Data in format (x, y)

        fmt: str, optional
        Default parameter. Not used

        compontnts: dict, optional
        Default parameter. Not used

        """
        _ = args, kwargs, fmt, components
        self.input = src[0][self.indices].reshape(-1, src[0].shape[1])
        self.labels = src[1][self.indices]
        return self
