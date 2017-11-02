""" Contains a base model class"""

class BaseModel:
    """ Base class for all models

    Attributes
    ----------
    name : str
        a model name
    config : dict
        configuration parameters

    Notes
    -----

    **Configuration**:

    * build : bool
        whether to build a model by calling `self.build()`. Default is True.
    * load : dict
        parameters for model loading. If present, a model will be loaded
        by calling `self.load(**config['load'])`.

    """
    def __init__(self, name=None, config=None, *args, **kwargs):
        self.config = config or {}
        self.name = name or self.__class__.__name__
        if self.get_from_config('build', True):
            self.build(*args, **kwargs)
        if self.get_from_config('load', False):
            self.load(**self.get_from_config('load'))

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
        _ = names
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
