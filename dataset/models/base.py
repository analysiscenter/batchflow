""" Contains a base model class"""

class BaseModel:
    """ Base model """
    def __init__(self, *args, **kwargs):
        self.config = kwargs.get('config', {})
        self.name = kwargs.get('name', None) or self.__class__.__name__
        if not kwargs.get('load', False):
            self.build(*args, **kwargs)

    def build(self, *args, **kwargs):
        """ Define the model """
        _ = self, args, kwargs

    def load(self, *args, **kwargs):
        """ Load the model """
        _ = self, args, kwargs

    def save(self, *args, **kwargs):
        """ Save the model """
        _ = self, args, kwargs

    def train(self, *args, **kwargs):
        """ Train the model """
        _ = self, args, kwargs

    def predict(self, *args, **kwargs):
        """ Make a prediction using the model  """
        _ = self, args, kwargs
