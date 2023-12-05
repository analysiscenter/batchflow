from pprint import pformat

class Config(dict):
    class IAddDict(dict):
        pass
    def __init__(self, config=None, **kwargs):
        self.config = {}

        if config is None:
            pass
        elif isinstance(config, Config):
            self.parse(config.config)
        elif isinstance(config, (dict, list, tuple)):
            self.parse(config)
        else:
            raise TypeError(f'Config must be dict, Config, list or tuple but {type(config)} was given')

        for key, value in kwargs.items():
            self.put(key, value)
    
    def parse(self, config):
        items = config.items() if isinstance(config, dict) else dict(config).items()

        for key, value in items:
            if isinstance(key, str):
                key = '/'.join(s for s in key.split('/') if s)
            self.put(key, value)

        return self

    def put(self, key, value):

        if isinstance(key, str):
            d = self.config
            levels = key.split('/')

            # Iterate to the last level
            for level in levels[:-1]:
                prev_d = d
                if level not in d:
                    d[level] = {}
                d = d[level]

            # Update the last leaf
            if isinstance(value, dict) and levels[-1] in d and isinstance(d[levels[-1]], dict):
                d[levels[-1]].update(value)
            else:
                if isinstance(d, dict):
                    d[levels[-1]] = value
                else:
                    prev_d[level] = {levels[-1]: value}
        else:
            self.config[key] = value

    def _get(self, key, default=None, has_default=False, pop=False):

        method = 'get' if not pop else 'pop'
        method = getattr(self.config, method)

        unpack = False
        if not isinstance(key, list):
            key = [key]
            unpack = True

        ret_vars = []
        for variable in key:
            if isinstance(variable, str) and '/' in variable:
                value = self.config
                levels = variable.split('/')
                values = []

                # Iterate to the last level
                for level in levels:
                    if not isinstance(value, dict):
                        if has_default:
                            return default
                        raise KeyError(level)
                    if level not in value:
                        if has_default:
                            return default
                        raise KeyError(level)
                    value = value[level]
                    values.append(value)
                if pop:
                    del values[-2][level]
            else:
                if variable not in self.config:
                    if has_default:
                        return default
                    raise KeyError
                value = method(variable)

            if isinstance(value, dict):
                value = Config(value)

            ret_vars.append(value)
            
        ret_vars = ret_vars[0] if unpack else tuple(ret_vars)
        return ret_vars

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
        value = self._get(key, default=default, has_default=True)

        return value
    
    def pop(self, key, **kwargs):
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
        has_default = 'default' in kwargs
        default = kwargs.get('default')
        value = self._get(key, has_default=has_default, default=default, pop=True)

        return value

    def __repr__(self):
        return repr(self.config)

    def __getitem__(self, key):
        value = self._get(key)
        return value
    
    def update(self, other=None, **kwargs):
        iterable = other if isinstance(other, (dict, tuple, list)) else kwargs
        
        if isinstance(iterable, dict):
            items = iterable.items()
        elif isinstance(iterable, (tuple, list)):
            items = iterable
            if any([not isinstance(item, (tuple, list)) for item in items]):
                raise ValueError('Tuple or list should contain only tuples or lists')
            if any([len(item) != 2 for item in items]):
                raise ValueError('Tuples in list should represent pairs key-value'
                                 ', and therefore must be always the length of 2')

        for key, value in iterable.items():
            self.put(key, value)
            
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
            keys = self.config.keys()
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
            values = self.config.values()
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
            items = self.config.items()
        return items
    
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
        config = self.config if config is None else config
        new_config = {}
        for key, value in config.items():
            if isinstance(value, dict) and len(value) > 0:
                value = self.flatten(value)
                for _key, _value in value.items():
                    if isinstance(_key, str):
                        new_config[key + '/' + _key] = _value
                    else:
                        new_config[key] = {_key: _value}
            else:
                new_config[key] = value

        return new_config
    
    def __setitem__(self, key, value):
        if key in self.config:
            self.pop(key, default=None)
        self.put(key, value)

    def __delitem__(self, key):
        self.pop(key)
    
    def copy(self):
        """ Create a shallow copy of the instance. """
        return Config(self.config.copy())
    
    def __getattr__(self, key):
        if key in self.config:
            value = self.config.get(key)
            value = Config(value) if isinstance(value, dict) else value
            return value
        raise AttributeError(key)
        
    def __add__(self, other):
        if isinstance(other, dict) and not isinstance(other, Config):
            other = Config(other)
        if isinstance(other, Config):
            return Config([*self.flatten().items(), *other.flatten().items()])
        return NotImplemented
    
    def __iter__(self):
        return iter(self.config)
    
    def __repr__(self):
        lines = ['\n' + 4 * ' ' + line for line in pformat(self.config).split('\n')]
        return f"Config({''.join(lines)})"
    
    def __iadd__(self, other):
        if isinstance(other, dict):
            self.update(other)
        else:
            raise TypeError(f"unsupported operand type(s) for +=: 'IAddDict' and '{type(other)}'")
        return self

    def __radd__(self, other):
        if isinstance(other, dict):
            other = Config(other)
        return other.__add__(self)
    
    def __len__(self):
        return len(self.config)
    
    def __eq__(self, other):
        self_ = self.flatten()
        print(self_, 'self_')
        other_ = Config(other).flatten() if isinstance(other, dict) else other
        return self_.__eq__(other_)

    def __getstate__(self):
        """ Must be explicitly defined for pickling to work. """
        return vars(self)

    def __setstate__(self, state):
        """ Must be explicitly defined for pickling to work. """
        vars(self).update(state)

    def __rshift__(self, other):
        """ Parameters
            ----------
            other : Pipeline

            Returns
            -------
            Pipeline
                Pipeline object with an updated config.
        """
        return other << self
