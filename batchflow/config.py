""" Config class"""
import warnings

class Config(dict):
    """ Class for configs that can be represented as nested dicts with easy indexing by slashes. """
    def __init__(self, config=None, **kwargs):
        """ Create Config

        Parameters
        ----------
        config : dict, Config, list, tuple or None
            An object to initialize Config.

            If dict, all keys with slashes and values are parsed into nested structure of dicts,
            and the resulting dictionary is saved to self.
            For example, `{'a/b': 1, 'c/d/e': 2}` will be parsed into `{'a': {'b': 1}, 'c': {'d': {'e': 2}}}`.

            If list or tuple, should contain key-value pairs with the length of 2.
            For example, `[('a/b', 1), ('c/d/e', 2)]` will be parsed into `{'a': {'b': 1}, 'c': {'d': {'e': 2}}}`.

            If an instance of Config, config is saved to self.

            If None, empty dictionary is created.
        kwargs :
            Parameters from kwargs also are parsed and saved to self.
        """
        if config is None:
            pass
        elif isinstance(config, Config):
            super().__init__(config)
        elif isinstance(config, (dict, list, tuple)):
            self.parse(config)
        else:
            raise TypeError(f'Config must be dict, Config, list or tuple but {type(config)} was given')

        for key, value in kwargs.items():
            self.put(key, value)

    def parse(self, config):
        """ Parses flatten config with slashes.

        Parameters
        ----------
        config : dict, Config, list or tuple

        Returns
        -------
        self : dict
        """
        if isinstance(config, dict):
            items = config.items()
        elif isinstance(config, (tuple, list)):
            items = config
            if any([not isinstance(item, (tuple, list)) for item in items]):
                raise ValueError('Tuple or list should contain only tuples or lists')
            if any([len(item) != 2 for item in items]):
                raise ValueError('Tuples in list should represent pairs key-value'
                                 ', and therefore must be always the length of 2')

        for key, value in items:
            if isinstance(key, str):
                key = '/'.join(filter(None, key.split('/')))
            self.put(key, value)

        return self

    def put(self, key, value):
        """ Put a new key into config recursively.

        Parameters
        ----------
        key : hashable object
            Key to add. '/' is used to put value into nested dict.
        value : misc
        """
        if isinstance(key, str) and '/' in key:
            config = self
            parent, child = key.split('/', 1)

            if parent in config and isinstance(config[parent], Config): # for example, we put value=3 with key='a/c' into the
                config[parent].update(Config({child: value}))           # config = {'a': {'b': 1}} and want to receive {'a': {'b': 1, 'c': 3}}
            else:
                if parent in config:
                    warnings.warn(f'Note that value for `{parent}` was overridden, not updated!')
                config[parent] = Config({child: value})

        else:
            if key in self and isinstance(self[key], dict) and isinstance(value, dict): 
                self[key].update(Config(value))
            else:
                super().__setitem__(key, value)
    
    def get(self, key, default=None):
        """ Returns the value or tuple of values for key in the config.
        If not found, returns a default value.

        Parameters
        ----------
        key : str or list of hashable objects
            A key in the dictionary. '/' is used to get value from nested dict.
        default : misc
            Default value if key doesn't exist in config.
            Defaults to None, so that this method never raises a KeyError.

        Returns
        -------
        Single value or a tuple.
        """
        value = self._get(key, default=default)
        return value
    
    def _get(self, key, pop=False, **kwargs):
        """ Recursively retrieve values for a given key if the key contains '/'.
        If key doesn't contain '/', get or pop from the `dict` class is invoked.

        Example:
        Let d = {'a': {'b': {'c': 30}}}.
        If we want to get d['a/b'], __getitem__ method will invoke _get method.

        Given the key='a/b', keys will be ['a', 'b'].
        value = self (value = {'a': {'b': {'c': 30}}}).
        k = 'a':
            Then parent starts to link to this value, i.e., parent = {'a': {'b': {'c': 30}}};
            Then we get value for 'a', so a new value starts to link to the {'b': {'c': 30}}.
        k = 'b':
            Then parent starts to link to the new value, i.e., parent = {'b': {'c': 30}};
            Then we get value for 'b', so a new value starts to link to the {'c': 30}.
        Returns {'c': 30}
        """
        has_default = 'default' in kwargs 
        default = kwargs.get('default')
        default = Config(default) if isinstance(default, dict) else default

        method = super().get if not pop else super().pop

        unpack = False
        if not isinstance(key, list):
            key = [key]
            unpack = True

        ret_vars = []
        for variable in key:
            if isinstance(variable, str) and '/' in variable:
                keys = variable.split('/') # split variable='a/b' into ['a', 'b']
                value = self # value starts to link to self which is original dict
                for k in keys:
                    if isinstance(value, dict):
                        parent = value # parent starts to link to value
                        value = value[k] # this invokes the __getitem__ method again and returns the value corresponding to the k,
                                         # value starts to link to the dict inside the previous dict

                    # if we want to get, for example, 'a/b/c' from {'a': {'b': 30}} 
                    else:
                        if has_default:
                            return default
                        raise KeyError(k)
                if pop:
                    del parent[k]

            # use dict's 'get' or 'pop' if '/' not in variable
            else:
                if variable in self:
                    value = method(variable)
                else:
                    if has_default:
                        return default
                    raise KeyError(variable)

            ret_vars.append(value)

        ret_vars = ret_vars[0] if unpack else tuple(ret_vars)

        return ret_vars
    
    def pop(self, key, default=None):
        """ Returns the value or tuple of values for key and remove them from config.

        Parameters
        ----------
        key : str or list of hashable objects
            A key in the dictionary. '/' is used to get value from nested dict.
        default : misc
            Default value if key doesn't exist in config.
            Defaults to None, so that this method never raises a KeyError.

        Returns
        -------
        Single value or a tuple
        """
        value = self._get(key, pop=True, default=default)
        return value
    
    def update(self, other, **kwargs):
        """ Update config with values from other.

        Parameters
        ----------
        other : dict or Config

        kwargs :
            parameters from kwargs also will be included into the resulting config
        """
        iterable = other if isinstance(other, dict) else kwargs
        for key, value in iterable.items():
            self.put(key, value)

    def flatten(self, config=None):
        """ Transforms nested dict into flatten dict.

        Parameters
        ----------
        config : dict, Config or None
            If None self will be parsed else config.

        Returns
        -------
        new_config : dict
        """
        config = self if config is None else config
        new_config = {}
        for key, value in config.items():
            if isinstance(value, dict) and len(value) > 0:
                value = self.flatten(value)
                for _key, _value in value.items():
                    new_config[key + '/' + _key] = _value
            else:
                new_config[key] = value

        return new_config
    
    def copy(self):
        """ Create a shallow copy of the instance. """
        return Config(super().copy())
    
    def keys(self, flatten=False):
        """ Returns config keys.

        Parameters
        ----------
        flatten : bool
            if False, keys will be getted from first level of nested dict, else from the last.

        Returns
        -------
            dict_keys
        """
        if flatten:
            keys = self.flatten().keys()
        else:
            keys = super().keys()
        return keys

    def values(self, flatten=False):
        """ Return config values.

        Parameters
        ----------
        flatten : bool
            if False, values will be getted from first level of nested dict, else from the last.

        Returns
        -------
            dict_values
        """
        if flatten:
            values = self.flatten().values()
        else:
            values = super().values()
        return values

    def items(self, flatten=False):
        """ Returns config items.

        Parameters
        ----------
        flatten : bool
            if False, keys and values will be getted from first level of nested dict, else from the last.

        Returns
        -------
            dict_items
        """
        if flatten:
            items = self.flatten().items()
        else:
            items = super().items()
        return items
    
    def __getitem__(self, key):
        value = self._get(key)
        return value

    def __setitem__(self, key, value):
        _ = self.pop(key, None)
        self.put(key, value)

    def __delitem__(self, key):
        self.pop(key)

    def __getattr__(self, key):
        if key in self:
            value = self.get(key)
            value = Config(value) if isinstance(value, dict) else value
            return value
        raise AttributeError(key)
    
    def __iadd__(self, other):
        if isinstance(other, dict):
            self.update(other)
        else:
            raise TypeError(f"Unsupported operand type(s) for +=: 'Config' and '{type(other)}'")
        return self

    def __add__(self, other):
        if isinstance(other, dict) and not isinstance(other, Config):
            other = Config(other)
        if isinstance(other, Config):
            return Config([*self.flatten().items(), *other.flatten().items()])
        return NotImplemented

    def __radd__(self, other):
        if isinstance(other, dict):
            other = Config(other)
        return other.__add__(self)

    def __eq__(self, other):
        self_ = self.flatten() if isinstance(self, Config) else self
        other_ = Config(other).flatten() if isinstance(other, dict) else other
        return self_.__eq__(other_)
    
    def __rshift__(self, other):
        """
            Parameters
            ----------
            other : Pipeline

            Returns
            -------
            Pipeline
                Pipeline object with an updated config.
        """
        return other << self
    
    def __getstate__(self):
        """ Must be explicitly defined for pickling to work. """
        return vars(self)

    def __setstate__(self, state):
        """ Must be explicitly defined for pickling to work. """
        vars(self).update(state)
