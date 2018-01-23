""" Config class"""

class Config:
    """ Class for configs that can be represented as nested dicts with easy indexing by slashes. """
    def __init__(self, config=None, **kwargs):
        if config is None:
            self.config = dict()
        elif isinstance(config, Config):
            self.config = config.config
        elif isinstance(config, dict):
            self.config = self.parse(config)
        self.update(**kwargs)

    def pop(self, variables, config=None, **kwargs):
        """ Return variables and remove them from config.

        Parameters
        ----------
        variables : str or list of strs
            names of variables. '/' is used to get value from nested dict.
        config : dict, Config or None
            if None variables will be getted from self.config.

        Returns
        -------
        single value or a tuple
        """
        if isinstance(config, Config):
            return config.pop(variables, None, **kwargs)
        else:
            return self._get(variables, config, pop=True, **kwargs)

    def get(self, variables, config=None, default=None):
        """ Return variables from config.

        Parameters
        ----------
        variables : str or list of strs
            names of variables. '/' is used to get value from nested dict.
        config : dict, Config or None
            if None variables will be getted from self.config.

        Returns
        -------
        single value or a tuple
        """
        if isinstance(config, Config):
            return config.get(variables, None, default)
        else:
            return self._get(variables, config, default=default, pop=False)

    def _get(self, variables, config=None, **kwargs):
        if config is None:
            config = self.config
        pop = kwargs.get('pop', False)
        has_default = 'default' in kwargs
        default = kwargs.get('default')

        unpack = False
        if not isinstance(variables, (list, tuple)):
            variables = list([variables])
            unpack = True

        ret_vars = []
        for variable in variables:
            _config = config
            if '/' in variable:
                var = variable.split('/')
                prefix = var[:-1]
                var_name = var[-1]
            else:
                prefix = []
                var_name = variable

            for p in prefix:
                if p in _config:
                    _config = _config[p]
                else:
                    _config = None
                    break
            if _config:
                if pop:
                    if has_default:
                        val = _config.pop(var_name, default)
                    else:
                        val = _config.pop(var_name)
                else:
                    if has_default:
                        val = _config.get(var_name, default)
                    else:
                        val = _config[var_name]
            else:
                if has_default:
                    val = default
                else:
                    raise KeyError("Key '%s' not found" % variable)

            ret_vars.append(val)

        if unpack:
            ret_vars = ret_vars[0]
        else:
            ret_vars = tuple(ret_vars)
        return ret_vars

    def put(self, variable, value, config=None):
        """ Put a new variable into config.

        Parameters
        ----------
        variable : str
            variable to add. '/' is used to put value into nested dict.
        value : masc
        config : dict, Config or None
            if None value will be putted into self.config.
        """
        if config is None:
            config = self.config
        elif isinstance(config, Config):
            config = config.config
        variable = variable.strip('/')
        if '/' in variable:
            var = variable.split('/')
            prefix = var[:-1]
            var_name = var[-1]
        else:
            prefix = []
            var_name = variable

        for p in prefix:
            if p not in config:
                config[p] = dict()
            config = config[p]
        if var_name in config and isinstance(config[var_name], dict) and isinstance(value, dict):
            config[var_name].update(value)
        else:
            config[var_name] = value

    def parse(self, config):
        """ Parse flatten config with slashes.
        Parameters
        ----------
        config : dict or Config

        Returns
        -------
        new_config : dict
        """
        if isinstance(config, Config):
            return config.config
        new_config = dict()
        for key, value in config.items():
            if isinstance(value, dict):
                value = self.parse(value)
            self.put(key, value, new_config)
        return new_config

    def flatten(self, config=None):
        """ Transform nested dict into flatten dict.
        Parameters
        ----------
        config : dict, Config or None
            if None self.config will be parsed else config.
        Returns
        -------
        new_config : dict
        """
        if config is None:
            config = self.config
        elif isinstance(config, Config):
            config = config.config
        new_config = dict()
        for key, value in config.items():
            if isinstance(value, dict):
                value = self.flatten(value)
                for _key, _value in value.items():
                    new_config[key+'/'+_key] = _value
            else:
                new_config[key] = value
        return new_config

    def __add__(self, other):
        if isinstance(other, dict):
            other = Config(other)
        return Config({**self.flatten(), **other.flatten()})

    def __radd__(self, other):
        if isinstance(other, dict):
            other = Config(other)
        return other.__add__(self)

    def __getitem__(self, key):
        value = self._get(key)
        return value

    def __setitem__(self, key, value):
        self.put(key, value)

    def __delitem__(self, key):
        self.pop(key)

    def __len__(self):
        return len(self.config)

    def items(self, flatten=False):
        """ Return config items. """
        if flatten:
            return self.flatten().items()
        else:
            return self.config.items()

    def keys(self, flatten=False):
        """ Return config keys. """
        if flatten:
            return self.flatten().keys()
        else:
            return self.config.keys()

    def values(self, flatten=False):
        """ Return config values. """
        if flatten:
            return self.flatten().values()
        else:
            return self.config.values()

    def update(self, other=None, **kwargs):
        """ Update config with values from other. """
        other = dict() if other is None else other
        if hasattr(other, 'keys'):
            for key in other:
                self[key] = other[key]
        else:
            for key, value in other:
                self[key] = value
        for key, value in kwargs.items():
            self[key] = value

    def __iter__(self):
        return iter(self.config)

    def __repr__(self):
        return "Config(" + str(self.config) + ")"
