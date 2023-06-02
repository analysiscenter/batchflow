""" Config class"""
from pathlib import Path
import numpy as np

class IAddDict(dict):
    """ dict that supports update via += """
    def __iadd__(self, other):
        if isinstance(other, dict):
            self.update(other)
        else:
            raise TypeError(f"unsupported operand type(s) for +=: 'IAddDict' and '{type(other)}'")
        return self

class Config(dict):
    """ Class for configs that can be represented as nested dicts with easy indexing by slashes """
    def __init__(self, config=None, **kwargs):
        """ Create Config

        Parameters
        ----------
        config : dict, Config or None
            an object to initialize Config
            if dict, all keys and values slashes will be parsed into nested structure of dicts
            and the resulting dictionary will be saved into self
            if an instance on Config, config will be saved to self
            if None, empty dictionary will be created
        kwargs :
            parameters from kwargs also will be parsed and saved into self
        """
        if config is None:
            pass
        elif isinstance(config, Config):
            super().__init__(config)
        elif isinstance(config, (dict, list)):
            self.parse(config)
        else:
            raise TypeError(f'config must be dict, Config or list but {type(config)} was given')

        for key, value in kwargs.items():
            self.put(key, value)

    def parse(self, config):
        """ Parses flatten config with slashes

        Parameters
        ----------
        config : dict, Config or list

        Returns
        -------
        new_config : dict
        """
        if isinstance(config, dict):
            items = config.items()
        elif isinstance(config, list):
            items = config
            if np.any([len(item) != 2 for item in items]):
                raise ValueError('tuples in list should represent pairs key-value'
                                 ', and therefore must be always the length of 2')

        for key, value in items:
            if not isinstance(key, (str, Path)):
                raise TypeError(f'only str and Path keys are supported, "{str(key)}" is of {type(key)} type')

            if isinstance(key, str):
                key = '/'.join(filter(None, key.split('/')))

            self.put(key, value)

        return self

    def put(self, key, value):
        """ Put a new variable into config

        Parameters
        ----------
        key : str, Path
            key to add. '/' is used to put value into nested dict
        value : masc
        """
        if not isinstance(value, Config) and isinstance(value, dict):
            value = Config(value)

        if isinstance(key, str) and '/' in key:
            keys = key.split('/')
            prefix = keys[:-1]
            var_name = keys[-1]

            config = self
            for i, p in enumerate(prefix):
                if p not in config:
                    config[p] = {}
                if isinstance(config[p], dict):
                    config = config[p]
                else:
                    value = Config({'/'.join(prefix[i+1:] + [var_name]): value})
                    var_name = p
                    break

            if var_name in config and isinstance(config[var_name], dict) and isinstance(value, Config):
                config[var_name].update(value)
            else:
                config[var_name] = value

        else:
            if key in self and isinstance(self[key], dict) and isinstance(value, Config):
                self[key].update(value)
            else:
                super().__setitem__(key, value)

    def get(self, variables, config=None, default=None):
        """ Returns variables from config

        Parameters
        ----------
        variables : str or list of str or tuple of str
            names of variables. '/' is used to get value from nested dict.
        config : dict, Config or None
            if None variables will be getted from self.config else from config
        default : masc
            default value if variable doesn't exist in config

        Returns
        -------
        single value or a tuple
        """
        if isinstance(config, Config):
            value = config.get(variables, default=default)
        else:
            value = self._get(variables, config=config, default=default)
        return value

    def _get(self, variables, config=None, **kwargs):
        pop = kwargs.get('pop', False)
        has_default = 'default' in kwargs
        default = kwargs.get('default')

        unpack = False
        if not isinstance(variables, (list, tuple)):
            variables = list([variables])
            unpack = True

        ret_vars = []
        for variable in variables:
            if isinstance(variable, str) and '/' in variable:
                keys = variable.split('/')
                prefix = keys[:-1]
                var_name = keys[-1]

                _config = self if config is None else config
                for p in prefix:
                    if p in _config:
                        _config = _config[p]
                    else:
                        _config = None
                        break

                if isinstance(_config, dict):
                    if pop:
                        value = _config.pop(var_name)
                    else:
                        value = _config[var_name]
                else:
                    if has_default:
                        value = default
                    else:
                        raise KeyError(f"Key '{variable}' not found")

            else:
                _config = self if config is None else config
                value = self._get_var_from_object(variable, has_default, default, pop, _config)

            ret_vars.append(value)

        if unpack:
            ret_vars = ret_vars[0]
        else:
            ret_vars = tuple(ret_vars)

        return ret_vars

    def _get_var_from_object(self, variable, has_default, default, pop, config):
        """ Get variable from the object.
        The object can be either Config or dict.
        If dict, the parent methods will be used.  
        """
        if isinstance(config, Config):
            obj = super()
        else:
            obj = config

        if variable in config:
            value = obj.pop(variable) if pop else obj.__getitem__(variable)
        else:
            if has_default:
                value = obj.pop(variable, default) if pop else obj.get(variable, default)
                value = Config(value) if isinstance(value, dict) and len(value) > 0 else value
            else:
                raise KeyError(f"Key '{variable}' not found")

        return value

    def update(self, other, **kwargs):
        """ Update config with values from other

        Parameters
        ----------
        other : dict or Config

        kwargs :
            parameters from kwargs also will be included into the resulting config
        """
        other = {} if other is None else other
        if isinstance(other, dict):
            for key, value in other.items():
                self.put(key, value)
        else:
            for key, value in kwargs.items():
                self.put(key, value)

    def pop(self, variables, config=None, default=None, **kwargs):
        """ Returns variables and remove them from config

        Parameters
        ----------
        variables : str or list of strs
            names of variables. '/' is used to get value from nested dict
        config : dict, Config or None
            if None, variables will be getted from self.config else from config

        Returns
        -------
        single value or a tuple
        """
        if isinstance(config, Config):
            value = config.pop(variables, default=default)
        else:
            value = self._get(variables, pop=True, default=default, **kwargs)
        return value

    def flatten(self, config=None):
        """ Transforms nested dict into flatten dict

        Parameters
        ----------
        config : dict, Config or None
            if None self.config will be parsed else config

        Returns
        -------
        new_config : dict
        """
        config = self if config is None else config

        new_config = IAddDict() # Do we really need here IAddDict?
        for key, value in config.items():
            if isinstance(value, dict) and len(value) > 0:
                value = self.flatten(value)
                for _key, _value in value.items():
                    new_config[key+'/'+_key] = _value
            else:
                new_config[key] = value

        return new_config

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
            raise TypeError(f"unsupported operand type(s) for +=: 'Config' and '{type(other)}'")
        return self

    def __add__(self, other):
        if isinstance(other, dict) and not isinstance(other, Config):
            other = Config(other)
        if isinstance(other, Config):
            return Config([*self.flatten().items(), *other.flatten().items()])
        return NotImplemented

    def __radd__(self, other):
        if isinstance(other, dict) and not isinstance(other, Config):
            other = Config(other)
        return other.__add__(self)

    def __setitem__(self, key, value):
        self.pop(key, default=None)
        self.put(key, value)

    def __getitem__(self, key):
        value = self._get(key)
        return value

    def __delitem__(self, key):
        self.pop(key)

    def __eq__(self, other):
        self_ = self.flatten() if isinstance(self, Config) else self
        other_ = Config(other).flatten() if isinstance(other, dict) and not isinstance(other, Config) else other
        return self_.__eq__(other_)

    def __rshift__(self, other):
        """
            Parameters
            ----------
            other : Pipeline

            Returns
            -------
            Pipeline
                Pipeline object with an updated config
        """
        return other << self

    def copy(self):
        """ Create a shallow copy of the instance. """
        return Config(super().copy())

    def __getstate__(self):
        """ Must be explicitly defined for pickling to work. """
        return vars(self)

    def __setstate__(self, state):
        """ Must be explicitly defined for pickling to work. """
        vars(self).update(state)

    def keys(self, flatten=False):
        """ Returns config keys

        Parameters
        ----------
        flatten : bool
            if False, keys will be getted from first level of nested dict, else from the last

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
        """ Return config values

        Parameters
        ----------
        flatten : bool
            if False, values will be getted from first level of nested dict, else from the last

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
        """ Returns config items

        Parameters
        ----------
        flatten : bool
            if False, keys and values will be getted from first level of nested dict, else from the last

        Returns
        -------
            dict_items
        """
        if flatten:
            items = self.flatten().items()
        else:
            items = super().items()
        return items