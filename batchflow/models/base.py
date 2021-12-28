""" Contains a base model class"""
from abc import ABC, abstractmethod

class BaseModel(ABC):
    """ Base interface for models. """

    @property
    def default_name(self):
        """ Placeholder for model name. """
        return self.__class__.__name__

    @abstractmethod
    def reset(self):
        """ Reset the trained model to allow a new training from scratch. """

    @abstractmethod
    def train(self):
        """ Train the model. """

    @abstractmethod
    def predict(self):
        """ Make a prediction using the model.  """

    @abstractmethod
    def load(self):
        """ Load the model from the disc. """

    @abstractmethod
    def save(self):
        """ Save the model to the disc. """

    @classmethod
    def is_model_like(cls, obj):
        """ Check if the `obj` provides the same interface, as required by this specification. """
        for method in cls.__abstractmethods__:
            if not hasattr(obj, method):
                return False
        return True
