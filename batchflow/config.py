from pprint import pformat

class Config(dict):

    # Should be defined temporarily for the already pickled configs
    class IAddDict(dict):
        pass

    def __init__(self, config=None, **kwargs):
        """ Create Config.

        Parameters
        ----------
        config : dict, Config, list, tuple or None
            An object to initialize Config.

            If dict, all keys with slashes and values are parsed into nested structure of dicts,
            and the resulting dictionary is saved to self.config.
            For example, `{'a/b': 1, 'c/d/e': 2}` will be parsed into `{'a': {'b': 1}, 'c': {'d': {'e': 2}}}`.

            If list or tuples, should contain key-value pairs with the length of 2.
            For example, `[('a/b', 1), ('c/d/e', 2)]` will be parsed into `{'a': {'b': 1}, 'c': {'d': {'e': 2}}}`.

            If an instance of Config, config is saved to self.config.

            If None, empty dictionary is created.
        kwargs :
            Parameters from kwargs also are parsed and saved to self.config.
        """
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
        """ Parses flatten config with slashes.

        Parameters
        ----------
        config : dict, Config, list or tuple

        Returns
        -------
        self : Config

        """
        if isinstance(config, Config):
            items = config.items(flatten=True) # suppose we have config = {'a': {'b': {'c': 1}}},
                                               # and we try to update config with other = {'a': {'b': {'d': 3}}},
                                               # and expect to see config = {'a': {'b': {'c': 1, 'd': 3}}}
        elif isinstance(config, dict):
            items = config.items()
        else:
            items = dict(config).items()
        # items = config.items() if isinstance(config, dict) else dict(config).items()

        for key, value in items:
            if isinstance(key, str): # if key contains multiple consecutive '/'
                key = '/'.join(s for s in key.split('/') if s)
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
        if isinstance(value, dict): # for example, value = {'a/b': 3}, and we need to parse it before put
            value = Config(value).config

        if isinstance(key, str):

            config = self.config
            levels = key.split('/')
            last_level = levels[-1]

            for level in levels[:-1]:
                prev_config = config
                if level not in config:
                    config[level] = {}
                config = config[level]

            if isinstance(value, dict) and last_level in config and isinstance(config[last_level], dict):
                config[last_level].update(value)
            else:
                # for example, we try to set config['a/b/c'] = 3, where config = Config({'a/b': 1}) and don't want error here
                if isinstance(config, dict):
                    config[last_level] = value
                else:
                    prev_config[level] = {last_level: value}
        else:
            self.config[key] = value

    def _get(self, key, default=None, has_default=False, pop=False):
        """ Consecutively retrieve values for a given key if the key contains '/'.
        This method supports the `default` to be unique for each variable in key.
        """
        method = 'get' if not pop else 'pop'
        method = getattr(self.config, method)

        unpack = False
        if not isinstance(key, list):
            key = [key]
            unpack = True

        # Provide `default` for each variable in key
        if default is not None and len(key) != 1 and len(default) != len(key):
            raise ValueError('You should provide `default` for each variable in `key`') # edit
        default = [default] if not isinstance(default, list) else default

        ret_vars = []
        for ix, variable in enumerate(key):

            if isinstance(variable, str) and '/' in variable:
                value = self.config
                levels = variable.split('/')
                values = []

                for level in levels:

                    if not isinstance(value, dict):
                        if not has_default:
                            raise KeyError(level)
                        value = default[ix]
                        ret_vars.append(value)
                        break

                    elif level not in value:
                        if not has_default:
                            raise KeyError(level)
                        value = default[ix]
                        ret_vars.append(value)
                        break

                    else:
                        value = value[level]
                        values.append(value)

                if pop:
                    del values[-2][level] # delete the last level from the parent dict

            else:
                if variable not in self.config:
                    if not has_default:
                        raise KeyError(variable)
                    value = default[ix]
                    ret_vars.append(value)

                else:
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
        value : misc
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
        value : misc
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
        other = other or {}
        if not isinstance(other, (dict, tuple, list)):
            raise TypeError(f'{type(other)} object is not iterable')

        self.parse(Config(other))

        for key, value in kwargs.items():
            self.put(key, value)

    def keys(self, flatten=False):
        """ Returns config keys

        Parameters
        ----------
        flatten : bool
            If False, keys will be got from first level of nested dict, else from the last.

        Returns
        -------
            keys : dict_keys

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
            If False, values will be got from first level of nested dict, else from the last.

        Returns
        -------
            values : dict_values

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
            If False, keys and values will be got from first level of nested dict, else from the last.

        Returns
        -------
            items : dict_items

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
            If None `self.config` will be parsed else config.

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
