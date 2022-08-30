""" Pipeline decorators """
import threading

from .named_expr import NamedExpression, eval_expr


class NonInitializedModel:
    """ Reference to a dynamic model that has not been created yet """
    def __init__(self, model_class, config=None, source=None):
        self.model_class = model_class
        self.config = config
        self.source = source

    @property
    def default_name(self):
        """: str - the model class name (serve as a default for a model name) """
        if isinstance(self.model_class, NamedExpression):
            raise ValueError("Model name should be explicitly set if a model class is a named expression",
                             self.model_class)
        return self.model_class.__name__


class ModelDirectory:
    """ Model storage """
    def __init__(self):
        self.models = {}
        self.lock = threading.Lock()

    def __getstate__(self):
        return {'models': self.models}

    def __setstate__(self, state):
        self.models = state['models']
        self.lock = threading.Lock()

    def __repr__(self):
        return repr(self.models)

    def copy(self):
        """ Make a shallow copy of the directory """
        new_md = ModelDirectory()
        new_md.models = {**self.models}
        return new_md

    def reset(self):
        """ Reset all models """
        for model in self.models.values():
            model.reset()

    def eval_expr(self, expr, batch=None):
        """ Evaluate all named expressions in a given data structure """
        return eval_expr(expr, batch=batch)

    def get(self, name):
        """ Retrieve a model from a directory without building it

        Parameters
        ----------
        name : str
            model name

        Returns
        -------
        model
        """
        return self.models.get(name)

    def get_model_by_name(self, name, batch=None):
        """ Retrieve a model from a directory

        Parameters
        ----------
        name : str
            model name

        Returns
        -------
        model

        Raises
        ------
        KeyError
            if there is no model with a given name
        """
        model = self.get(name)
        if model is None:
            raise KeyError(f"Model '{name}' does not exist")
        if isinstance(model, NonInitializedModel):
            with self.lock:
                model = self.get(name)
                if isinstance(model, NonInitializedModel):
                    source = self.eval_expr(model.source, batch=batch)
                    if source is None:
                        config = self.eval_expr(model.config, batch=batch) or {}
                        model_class = self.eval_expr(model.model_class, batch=batch)
                        model = self.create_model(model_class, config)
                        self.models.update({name: model})
                    else:
                        self.import_model(name, source, lock=False)
                        model = self.models[name]
        return model

    def create_model(self, model_class, config=None):
        """ Create a model """
        model = model_class(config=config)
        return model

    def add_model(self, name, model):
        """ Add a model to the directory """
        if name is None:
            name = model.default_name
        with self.lock:
            self.models.update({name: model})

    def init_model(self, name=None, model_class=None, mode='dynamic', *args, config=None, source=None):
        """ Initialize a static or dynamic model

        Parameters
        ----------
        name : str
            a name for the model (to refer to it later when training or infering).

        model_class : class or named expression
            a model class (might also be specified in the config).

        mode : {'static', 'dynamic'}
            model creation mode:
            - static - the model is created right now, during the pipeline definition
            - dynamic - the model is created at the first iteration when the pipeline is run (default)

        config : dict or Config
            model configurations parameters, where each key and value could be named expressions.

        source
            a model or a pipeline to import from
        """
        _ = args
        # workaround for a previous arg order
        if name in ['dynamic', 'static']:
            raise DeprecationWarning('Arguments order has changed to <model name>, <model class>, <mode>, <config>.')

        if model_class is None and config is not None:
            model_class = config.get('model_class')
        if model_class is None and source is None:
            raise ValueError('model_class should be specified in the model config')

        if mode == 'static':
            model = self.create_model(model_class, config)
        else:
            model = NonInitializedModel(model_class, config, source)
        self.add_model(name, model)

    def import_model(self, name, source, lock=True):
        """ Import model from another pipeline or a model itself """
        if isinstance(getattr(source, 'models', None), ModelDirectory):
            # so source is a pipeline (checking for it would cause cyclic import)
            model = source.m(name)
        else:
            model = source
        if lock:
            self.add_model(name, model)
        else:
            self.models.update({name: model})

    def save_model(self, name, *args, **kwargs):
        model = self.get_model_by_name(name)
        model.save(*args, **kwargs)

    def load_model(self, name, model_class=None, mode='dynamic', *args, build=None, **kwargs):
        """ Load a model """
        _ = args
        config = {'load': kwargs, 'build': build}
        self.init_model(name, model_class, mode, config=config)

    def __add__(self, other):
        if not isinstance(other, ModelDirectory):
            raise TypeError(f"ModelDirectory is expected, but given '{type(other).__name__}'")

        new_md = self.copy()
        new_md.models.update(other.models)
        return new_md
