""" Config class"""

class Config:
    """ Class for configs that can be represented as nested dicts with easy indexing by slashes """
    def __init__(self, config=None, **kwargs):
        """ Create Config

        Parameters
        ----------
        config : dict, Config or None
            an object to initialize Config
            if dict, all keys and values slashes will be parsed into nested structure of dicts
            and the resulting dictionary will be saved into self.config
            if an instance on Config, config.config will be saved to self.config (not a copy!)
            if None, empty dictionary will be created
        kwargs :
            parameters from kwargs also will be parsed and saved into self.config
        """
        if config is None:
            self.config = dict()
        elif isinstance(config, dict):
            self.config = self.parse(config)
        else:
            self.config = config.config
        for key, value in kwargs.items():
            self.put(key, value)

    def pop(self, variables, config=None, **kwargs):
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
            return config.pop(variables, None, **kwargs)
        else:
            return self._get(variables, config, pop=True, **kwargs)

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
            val = config.get(variables, default=default)
        else:
            val = self._get(variables, config, default=default, pop=False)
        return val

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
            if isinstance(_config, dict):
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
        """ Put a new variable into config

        Parameters
        ----------
        variable : str
            variable to add. '/' is used to put value into nested dict
        value : masc
        config : dict, Config or None
            if None value will be putted into self.config else from config
        """
        if config is None:
            config = self.config
        elif isinstance(config, Config):
            config = config.config
        if isinstance(value, dict):
            value = Config(value)
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
        if var_name in config and isinstance(config[var_name], dict) and isinstance(value, Config):
            config[var_name] = Config(config[var_name])
            config[var_name].update(value)
            config[var_name] = config[var_name].config
        else:
            if isinstance(value, Config):
                config[var_name] = value.config
            else:
                config[var_name] = value

    def parse(self, config):
        """ Parses flatten config with slashes

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
        """ Transforms nested dict into flatten dict

        Parameters
        ----------
        config : dict, Config or None
            if None self.config will be parsed else config

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
            if isinstance(value, Config):
                value = value.config
            if isinstance(value, dict) and len(value) > 0:
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
            return self.flatten().items()
        else:
            return self.config.items()

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
            return self.flatten().keys()
        else:
            return self.config.keys()

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
            return self.flatten().values()
        else:
            return self.config.values()

    def update(self, other=None, **kwargs):
        """ Update config with values from other

        Parameters
        ----------
        other : dict or Config

        kwargs :
            parameters from kwargs also will be included into the resulting config
        """
        other = dict() if other is None else other
        if isinstance(other, Config):
            new_config = self + other
            self.config = new_config.config
        elif isinstance(other, dict):
            new_config = self + Config(other)
            self.config = new_config.config
        else:
            for key, value in other:
                self.put(key, value)
        for key, value in kwargs.items():
            self.put(key, value)

    def __iter__(self):
        return iter(self.config)

    def __repr__(self):
        return "Config(" + str(self.config) + ")"
