""" Contains a base model class"""

class BaseModel:
    """ Base model

    Attributes
    ----------
    name : str - a model name
    config : dict - configuration parameters

    Configuration
    -------------
    build : bool - whether to build a model by calling `self.build()`. Default is True.
    load : bool - whether to load a model by calling `self.load()`. Default is False.
    """
    def __init__(self, *args, **kwargs):
        self.config = kwargs.get('config', {})
        self.name = kwargs.get('name', None) or self.__class__.__name__
        if self.get_from_config('build', True):
            self.build(*args, **kwargs)
        if self.get_from_config('load', False):
            self.load(**self.config)

    def get_from_config(self, variable, default=None):
        """ Return a variable from config or a default value """
        return self.config.get(variable, default)

    def _make_inputs(self, names=None):
        """ Make model input data using config

        Parameters
        ----------
        names : a sequence of str - names for input variables

        Returns
        -------
        None or dict - where key is a variable name and a value is a corresponding variable after configuration
        """
        return None

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
