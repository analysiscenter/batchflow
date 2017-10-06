""" Contains a base model class"""

class BaseModel:
    """ Base model """
    def __init__(self, *args, **kwargs):
        self.config = kwargs.get('config', None)
        self.build(*args, **kwargs)

    def build(self, *args, **kwargs):
        """ Define the model """
        return None

    def load(self, *args, **kwargs):
        """ Load the model """
        return None

    def save(self, *args, **kwargs):
        """ Save the model """
        return None

    def train(self, *args, **kwargs):
        """ Train the model """
        return None

    def predict(self, *args, **kwargs):
        """ Make a prediction using the model  """
        return None
