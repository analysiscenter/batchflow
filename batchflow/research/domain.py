""" Domain of parameters to generate configs. """

from itertools import product, islice
from collections import OrderedDict
from copy import copy, deepcopy
from pprint import pformat
import numpy as np

from .utils import must_execute
from ..utils import to_list
from .. import Config, Sampler, make_rng
from ..named_expr import eval_expr

class Alias:
    """ Class to create alias for some Python object. This is useful for creating short names for complex objects
    such as nested dictionaries.

    Parameters
    ----------
    value : object

    alias : str, optional
        Alias for value, by default None. If None then alias will be equal to `value.__name__`
        (if exists) or to `str(value)`.
    """
    def __init__(self, value, alias=None):
        if isinstance(value, Alias):
            self.value = value.value
            self.alias = value.alias
        else:
            self.value = value
            if alias is None:
                self.alias = self._get_name(value)
            else:
                self.alias = alias

    def __repr__(self):
        return 'Alias(' + str(self.alias) + ': ' + str(self.value) + ')'

    def _get_name(self, value):
        """ Create name for the value. """
        if hasattr(value, '__name__'):
            return value.__name__
        return str(value)

class ConfigAlias:
    """ Wrapper for Config to infer its aliased version. Each key and value from initial config will be
    wrapped with `Alias` class (if it is not).

    Parameters
    ----------
    config : dict, list of tuple
        each tuple is a pair (key, value), key is `Alias` or str, value is `Alias` or object.

    Notes
    -----
    ConfigAlias has two main methods: `config` and `alias`. `config` returns initial config as `Config` instance.
    `alias` returns aliased versions of config or its string representation.
    """
    def __init__(self, config=None):
        if isinstance(config, ConfigAlias):
            _config = config._config
        else:
            _config = []
            if isinstance(config, (dict, Config)):
                config = config.items()
            if config is not None:
                for key, value in config:
                    _key = key if isinstance(key, Alias) else Alias(key)
                    _value = value if isinstance(value, Alias) else Alias(value)
                    _config.append((_key, _value))
        self._config = _config

    def alias(self, as_string=False, delim='-'):
        """ Returns config alias.

        Parameters
        ----------
        as_string : bool, optional
            if True, return string representation of ConfigAlias. Different items will be
            separated by `delim`, key and value for each pair will be separated by '_'.
        delim : str, optional
            delimiter for different ConfigAlias items in string representation.

        Returns
        -------
        dict or str
        """
        config_alias = Config({item[0].alias: item[1].alias for item in self._config})
        if as_string:
            config_alias = OrderedDict(sorted(config_alias.items()))
            config_alias = delim.join([str(key)+'_'+str(value) for key, value in config_alias.items()])
        return config_alias

    def config(self):
        """ Returns initial config as `Config` instance.

        Returns
        -------
            Config
        """
        return Config({item[0].value: item[1].value for item in self._config})

    def pop_config(self, key):
        """ Pop item from ConfigAlias by config value (not by alias).

        Returns
        -------
        ConfigAlias or None
            ConfigAlias for popped keys. None if key doesn't exist.
        """
        key = to_list(key)
        res = [item for item in self._config if item[0].value in key]
        self._config = [item for item in self._config if item[0].value not in key]
        if len(res) >= 1:
            return ConfigAlias(res)
        return None

    def pop_alias(self, key):
        """ Pop item from ConfigAlias by alias (not by value).

        Returns
        -------
        ConfigAlias or None
            ConfigAlias for popped keys. None if key doesn't exist.
        """
        key = to_list(key)
        res = [item for item in self._config if item[0].alias in key]
        self._config = [item for item in self._config if item[0].alias not in key]
        if len(res) >= 1:
            return ConfigAlias(res)
        return None

    def set_prefix(self, keys, n_digits):
        """ Create prefix from keys. """
        prefix = ''
        for key in keys:
            prefix += self.alias().get('#' + key, 'null') + '_'
        fmt = ("{:0" + str(n_digits) + "d}").format(self.config()['repetition'])
        self['_prefix'] = prefix + fmt + '_'
        return self

    def __getitem__(self, key):
        """ Returns true value (not alias). """
        return self.config()[key]

    def __setitem__(self, key, value):
        _key = key if isinstance(key, Alias) else Alias(key)
        _value = value if isinstance(value, Alias) else Alias(value)
        self._config.append((_key, _value))

    def __repr__(self):
        return pformat(self.alias().config)

    def __add__(self, other):
        config = ConfigAlias()
        config._config = deepcopy(self._config) + deepcopy(other._config)
        return config

    def keys(self):
        return self.config().keys()

class Domain:
    """ Domain of parameters to generate configs for experiments.

    Parameters
    ----------
    domain : dict
        parameter values to try. Each key is a parameter, values is a list of parameter values
        or batchflow.Sampler.

    **kwargs :
        the same as a `domain` dict. `domain` using is preferable when parameter name includes symbols like `'/'`.

    Note
    ----
    `Domain` generates configs of parameters. The simplest example is `Domain(a=[1,2,3])`. That domain defines
    parameter `'a'` and its possible values `[1,2,3]`. You can iterate over all possible configs (3 configs in our
    example) and repeat generated configs in the same order several times (see `n_reps` in :meth:`~.set_iter_params`).

    Besides, parameter values can be a `batchflow.Sampler`, e.g. `Domain(a=NumpySampler('normal'))`. In that case
    values for parameter `'a'` will be sampled from normal distribution.

    Dict in domain definition can consist of several elements, then we will get all possible combinations of parameters,
    e.g. `Domain(a=[1,2], b=[3,4])` will produce four configs. If domain has parameters with array-like values and
    with sampler as values simultaneously, domain will produce all possible combinations of parameters with array-like
    values and for each combination values of other parameters will be sampled.

    To get configs from `Domain` use :meth:`~.iterator`. It produces configs wrapped by :class:`~.ConfigAlias`.

    Additional parameters like the number of repetitions or the number of samples for domains with samplers
    are defined in :meth:`~.set_iter_params`.

    **Operations with Domain**

    #. sum by `+`: Concatenate two domains. For example, the resulting domain
       `Domain(a=[1]) + Domain(b=[1])` will produce two configs: `{'a': 1}`, `{'b': 1}`
       (not one dict with `'a'` and `'b'`).

    #. multiplication by `*`: Cartesian multiplications of options in Domain.
       For example, if `domain1 = Domain({'a': [1, 2]})`, `domain2 = Domain({'b': [3, 4]})` and
       `domain3 = Domain({'c': bf.Sampler('n')})` then `domain1 * domain2 * domain3` will have
       all options and generate 4 configs: `{'a': 1, 'b': 3, 'c': xi_1}`, `{'a': 1, 'b': 4, 'c': xi_2}`,
       `{'a': 2, 'b': 3, 'c': xi_3}`, `{'a': 2, 'b': 4, 'c': xi_4}` where xi_i are independent samples from
       normal distribution. The same resulting domain can be defined as `Domain({'a': [1, 2], 'b': [3, 4],
       'c': bf.Sampler('n')})`.
    #. multiplication by @: element-wise multiplication of array-like options.
       For example, if `domain1 = Domain({'a': [1, 2]})` and `domain2 = Domain({'b': [3, 4]})` then
       `domain1 @ domain2` will have two configs:
       `{'a': 1, `b`: 3}`, `{'a': 2, `b`: 4}`.
    #. multiplication with weights: can be used to sample configs from sum of domains.
       For example, the first ten configs from `0.3 * Domain({'p1': NS('n', loc=-10)}) + 0.2 * Domain({'p2':  NS('u')})
       + 0.5 * Domain({'p3':  NS('n', loc=10)})` will be `{'p1': -10.3059}, {'p3': 8.9959},
       {'p3': 9.1302}, {'p3': 10.2611}, {'p1': -7.9388}, {'p2': 0.5455}, {'p1': -9.2497},
       {'p3': 9.9769}, {'p2': 0.3510}, {'p3': 8.8519}` (depends on seed).

    If you sum options with and without weights, they are grouped into consequent groups where all options has or
    not weights, for each group configs are generated consequently (for groups with weights) or sampled as described
    above. For example, for `domain = domain1 + 1.2 * domain2 + 2.3 * domain3 + domain4 + 1. * domain5` we will get:

        - all configs from domain1
        - configs will be sampled from 1.2 * domain2 + 2.3 * domain3
        - all configs from domain4
        - configs will be sampled from 1. * domain4

    If one of the domains here is a sampler-like domain, then samples from that domain will be generated endlessly.
    """
    def __init__(self, domain=None, **kwargs):
        if isinstance(domain, dict):
            self.cubes = [self.create_aliases(domain)]
            self.weights = np.array([np.nan])
        elif isinstance(domain, list) and all(isinstance(item, list) for item in domain):
            self.cubes = domain
            self.weights = np.array([np.nan] * len(domain))
        elif isinstance(domain, Domain):
            self.cubes = copy(domain.cubes)
            self.weights = copy(domain.weights)
        elif len(kwargs) > 0:
            self.cubes = [self.create_aliases(kwargs)]
            self.weights = np.array([np.nan])
        elif domain is None:
            self.cubes = []
            self.weights = np.array([])
        else:
            raise ValueError(f'domain can be Domain, dict or nested list but {type(domain)} were given')

        self.updates = []
        self.n_produced = 0

        self._iterator = None
        self.n_items = None
        self.n_reps = 1
        self.repeat_each = None
        self.n_updates = 0
        self.additional = True
        self.create_id_prefix = False
        self.random_state = None

        self.values_indices = dict()

    def _get_all_options_names(self):
        options = []
        for cube in self.cubes:
            for option in cube:
                alias = option[0].alias
                if alias not in options and alias != 'repetition':
                    options.append(alias)
        return options

    def create_aliases(self, options):
        """ Create aliases by wrapping into Alias class for each key and value of the dict. """
        aliases_options = []
        for parameter, values in options.items():
            parameter = Alias(parameter)
            if isinstance(values, (list, tuple, np.ndarray)):
                values = [Alias(value) for value in values]
            elif isinstance(values, Sampler):
                pass
            else:
                raise TypeError(f'`values` must be array-like object or Sampler but {type(values)} were given')
            aliases_options += [(parameter, values)]
        return aliases_options

    def set_iter_params(self, n_items=None, n_reps=1, repeat_each=None, produced=0, additional=True,
                        create_id_prefix=False, seed=None):
        """ Set parameters for iterator.

        Parameters
        ----------
        n_items : int or None
            the number of configs that will be generated from domain. If the size
            of domain is less then `n_items`, elements will be repeated. If `n_items`
            is `None` and there is not a cube that consists only of sampler-options
            then `n_items` will be setted to the number of configs that can be produced
            from that domain. If `n_items` is None and there is a cube that consists
            only of sampler-option then domain will produce infinite number of configs.
        n_reps : int
            each element will be repeated `n_reps` times.
        repeat_each : int
            if there is not a cube that consists only of sampler-options then
            elements will be repeated after producing `repeat_each` configs. Else
            `repeat_each` will be setted to the number of configs that can be produced
            from domain.
        produced : int
            how many configs was produced before (is needed to use after domain update).
        additional : bool
            append 'repetition' and 'updates' to config or not.
        seed : bool or int or object with a seed sequence attribute
            see :meth:`~batchflow.utils_random.make_seed_sequence`.
        """
        n_configs = self.len # None means that domain has samplers
        self.n_items = n_items or n_configs
        self.n_reps = n_reps
        if self.n_items is not None:
            self.repeat_each = repeat_each or self.n_items
        else:
            self.repeat_each = repeat_each or 100
        self.n_produced = produced
        self.additional = additional
        self.create_id_prefix = create_id_prefix
        self.random_state = make_rng(seed)
        self.reset_iter()

    def set_update(self, function, when, **kwargs):
        """ Set domain update parameters. """
        if isinstance(when, (int, str)):
            when = [when]
        iter_kwargs = dict()
        for attr in ['n_items', 'n_reps', 'repeat_each']:
            iter_kwargs[attr] = kwargs.pop(attr) if attr in kwargs else getattr(self, attr)
        self.updates.append({
            'function': function,
            'when': when,
            'kwargs': kwargs,
            'iter_kwargs': iter_kwargs
        })

    def update(self, generated, research):
        """ Update domain by `update_func`. If returns None, domain will not be updated. """
        for update in self.updates:
            if must_execute(generated-1, update['when'], self.n_produced + self.size):
                kwargs = eval_expr(update['kwargs'], research=research)
                domain = update['function'](**kwargs)
                domain.updates = self.updates
                domain.n_updates = self.n_updates + 1
                domain.values_indices = self.values_indices
                domain.set_iter_params(produced=generated, additional=self.additional, seed=self.random_state,
                                       create_id_prefix=self.create_id_prefix, **update['iter_kwargs'])
                return domain
        return None

    @property
    def size(self):
        """ Return the number of configs that will be produces from domain. """
        if self.n_items is not None:
            return self.n_reps * self.n_items
        return None

    @property
    def len(self):
        """ Return the number of configs that will be produced from domain without repetitions. None if infinite. """
        size = 0
        for cube in self.cubes:
            lengthes = [len(values) for _, values in cube if isinstance(values, (list, tuple, np.ndarray))]
            if len(lengthes) == 0:
                return None
            size += np.product(lengthes)
        return size

    def __len__(self):
        """ __len__ can't return None so we have to separate functions. """
        cube_sizes = [
            np.prod([len(values) for _, values in cube if isinstance(values, (list, tuple, np.ndarray))], dtype='int')
            for cube in self.cubes
        ] # np.prod returns 1.0 for empty list
        return max(0, sum(cube_sizes))

    def __mul__(self, other):
        if isinstance(other, float) and np.isnan(other):
            return self
        if self.cubes is None:
            result = other
        elif isinstance(other, (int, float)):
            result = self
            weights = self.weights
            weights[np.isnan(weights)] = 1
            result.weights = weights * other
        elif isinstance(other, Domain):
            if other.cubes is None:
                result = self
            else:
                res = list(product(self.cubes, other.cubes))
                res = [item[0] + item[1] for item in res]
                pairs = np.array(list(product(self.weights, other.weights)))
                weights = np.array([np.nanprod(item) for item in pairs])
                nan_mask = np.array([np.isnan(item).all() for item in pairs])
                weights[nan_mask] = np.nan
            result = Domain()
            result.cubes = res
            result.weights = weights
        else:
            raise TypeError('Arguments must be numeric or Domains')
        return result

    def __matmul__(self, other):
        if self._is_array_option():
            that = self._to_scalar_product()
        else:
            that = self

        if other._is_array_option():
            other = other._to_scalar_product()

        if that._is_scalar_product() and other._is_scalar_product():
            if len(that.cubes) == len(other.cubes):
                cubes = [cube_1 + cube_2 for cube_1, cube_2 in zip(that.cubes, other.cubes)]
                weights = np.nanprod(np.stack([that.weights, other.weights]), axis=0)
                nan_mask = np.logical_and(np.isnan(that.weights), np.isnan(other.weights))
                weights[nan_mask] = np.nan
                domain = Domain()
                domain.cubes = cubes
                domain.weights = weights
                return domain
        raise ValueError("The numbers of domain cubes must conincide.")

    def __rmul__(self, other):
        return self * other

    def __add__(self, other):
        if self.cubes is None:
            result = other
        elif other.cubes is None:
            result = self
        else: # Domain
            result = Domain()
            result.cubes = self.cubes + other.cubes
            result.weights = np.concatenate((self.weights, other.weights))
        return result

    def __getitem__(self, index):
        domain = Domain()
        domain.cubes = [self.cubes[index]]
        return domain

    def __eq__(self, other):
        return self.cubes == other.cubes

    def __next__(self):
        return next(self.iterator)

    def reset_iter(self):
        """ Reset iterator and set seeds for samplers. """
        for cube in self.cubes:
            for _, values in cube:
                if isinstance(values, Sampler):
                    values.state = make_rng(self.random_state)
        self._iterator = None

    def create_iter(self):
        """ Create iterator. """
        blocks = self._get_sampling_blocks()
        keys = self._get_all_options_names()

        def _iterator():
            while True:
                for block in blocks:
                    weights = self.weights[block]
                    weights[np.isnan(weights)] = 1
                    iterators = [self._cube_iterator(cube) for cube in np.array(self.cubes, dtype=object)[block]]
                    while len(iterators) > 0:
                        index = self.random_state.choice(len(block), p=weights/weights.sum())
                        try:
                            yield next(iterators[index])
                        except StopIteration:
                            del iterators[index]
                            weights = np.delete(weights, index)
                            block = np.delete(block, index)

        def _iterator_with_repetitions():
            iterator = _iterator()
            if self.n_reps == 1:
                i = 0
                if self.additional:
                    additional = ConfigAlias([('repetition', 0)]) + ConfigAlias([('updates', self.n_updates)])
                else:
                    additional = ConfigAlias()
                while self.n_items is None or i < self.n_items:
                    res = next(iterator) + additional # pylint: disable=stop-iteration-return
                    if self.create_id_prefix:
                        res.set_prefix(keys, n_digits=int(self.create_id_prefix))
                    yield res
                    i += 1
            else:
                i = 0
                while self.n_items is None or i < self.n_items:
                    samples = list(islice(iterator, int(self.repeat_each)))
                    for rep in range(self.n_reps):
                        if self.additional:
                            additional = ConfigAlias({'repetition': rep}) + ConfigAlias({'updates': self.n_updates})
                        else:
                            additional = ConfigAlias()
                        for sample in samples:
                            res = sample + additional
                            if self.create_id_prefix:
                                res.set_prefix(keys, n_digits=int(self.create_id_prefix))
                            yield res
                    i += self.repeat_each
        self._iterator = _iterator_with_repetitions()

    def _get_sampling_blocks(self):
        """ Return groups of cubes. Cubes are split into consequent groups where all cubes has or not weights. """
        incl = np.cumsum(np.isnan(self.weights))
        excl = np.concatenate(([0], incl[:-1]))
        block_indices = incl + excl
        return [np.where(block_indices == i)[0] for i in set(block_indices)]

    @property
    def iterator(self):
        """ Get domain iterator. """
        if self._iterator is None:
            self.set_iter_params(self.n_items, self.n_reps, self.repeat_each, self.n_produced,
                                 self.additional, self.create_id_prefix, self.random_state)
            self.create_iter()
        return self._iterator

    def _is_array_option(self):
        """ Return True if domain consists of only one array-like option. """
        if len(self.cubes) == 1:
            if len(self.cubes[0]) == 1:
                if isinstance(self.cubes[0][0][1], (list, tuple, np.ndarray)):
                    return True
        return False

    def _is_scalar_product(self):
        """ Return True if domain is a result of matmul. It means that each cube has
        an only one array-like option of length 1.
        """
        for cube in self.cubes:
            samplers = [name for name, values in cube if isinstance(values, Sampler)]
            if len(samplers) > 0:
                return False
            if any(len(values) != 1 for _, values in cube):
                return False
        return True

    def _to_scalar_product(self):
        """ Transform domain to the matmul format (see :meth:`~.Domain._is_scalar_product`)"""
        if self._is_array_option():
            name, values = self.cubes[0][0]
            cubes = [[[name, [value]]] for value in values]
            weights = np.concatenate([[self.weights[0]] * len(cubes)])
            domain = Domain()
            domain.cubes = cubes
            domain.weights = weights
            return domain
        if self._is_scalar_product():
            return Domain(self)
        raise ValueError("Domain cannot be represented as scalar product.")

    def _cube_iterator(self, cube):
        """ Return iterator from the cube. All array-like options will be transformed
        to Cartesian product and all sampler-like options will produce independent samples
        for each condig. """
        arrays = [item for item in cube if isinstance(item[1], (list, tuple, np.ndarray))]
        samplers = [item for item in cube if isinstance(item[1], Sampler)]

        if len(arrays) > 0:
            for combination in list(product(*[self.option_items(name, values) for name, values in arrays])):
                res = []
                for name, values in samplers:
                    res.append(self.option_sample(name, values))
                res.extend(combination)
                yield sum(res, ConfigAlias())
        else:
            iterators = [self.option_iterator(name, values) for name, values in cube]
            while True:
                try:
                    yield sum([next(iterator) for iterator in iterators], ConfigAlias())
                except StopIteration:
                    break

    def option_items(self, name, values):
        """ Return all possible `ConfigAlias` instances which can be created from the option.

        Returns
        -------
        list of `ConfigAlias` objects.
        """
        if not isinstance(values, (list, tuple, np.ndarray)):
            raise TypeError(f'`values` must be array-like object but {type(values)} were given')
        res = []
        for value in values:
            if self.create_id_prefix:
                n_digits = self.create_id_prefix if self.create_id_prefix is not True else 1
                option_values = self.values_indices.get(name.alias, dict())
                current_index = option_values.get(value.alias, len(option_values))
                option_values[value.alias] = current_index
                self.values_indices[name.alias] = option_values
                fmt = ("{:0" + str(n_digits) + "d}").format(current_index)
                res.append(ConfigAlias([[name, value], ["#" + name.alias, fmt]]))
            else:
                res.append(ConfigAlias([[name, value]]))
        return res

    def option_sample(self, name, values, size=None):
        """ Return `ConfigAlias` objects created on the base of Sampler-option.

        Parameters
        ----------
        size : int or None
            the size of the sample

        Returns
        -------
            ConfigAlias (if size is None) or list of ConfigAlias objects (otherwise).
        """

        if not isinstance(values, Sampler):
            raise TypeError(f'`values` must be Sampler but {type(values)} was given')
        res = []
        for _ in range(size or 1):
            if self.create_id_prefix:
                n_digits = self.create_id_prefix if self.create_id_prefix is not True else 1
                current_index = self.values_indices.get(name.alias, -1) + 1
                self.values_indices[name.alias] = current_index
                fmt = ("{:0" + str(n_digits) + "d}").format(current_index)
                res.append(ConfigAlias([[name, values.sample(1)[0, 0]], ["#" + name.alias, fmt]]))
            else:
                res.append(ConfigAlias([[name, values.sample(1)[0, 0]]]))
        if size is None:
            res = res[0]
        return res

    def option_iterator(self, name, values):
        """ Produce `ConfigAlias` from the option.

        Returns
        -------
            generator.
        """
        if isinstance(values, Sampler):
            while True:
                yield ConfigAlias([[name, values.sample(1)[0, 0]]])
        else:
            for value in values:
                yield ConfigAlias([[name, value]])

    def __repr__(self):
        repr = ''
        cubes_reprs = []
        spacing = 4 * ' '

        for cube in self.cubes:
            cubes_reprs += [' * '.join([self._option_repr(name, values) for name, values in cube])]
        repr += ' + \n'.join(cubes_reprs)
        repr += 2 * '\n' + 'params:\n'
        repr += '\n'.join([spacing + f"{attr}={getattr(self, attr)}" for attr in ['n_items', 'n_reps', 'repeat_each']])

        if len(self.updates) > 0:
            repr += 2 * '\n' + 'updates:\n'
            update_reprs = []
            for update in self.updates:
                update_reprs += [str('\n'.join(spacing + f"{key}: {value}" for key, value in update.items()))]
            repr += '\n\n'.join(update_reprs)
        return repr

    def _option_repr(self, name, values):
        alias = name.alias

        if isinstance(values, (list, tuple, np.ndarray)):
            values = [item.alias if not isinstance(item.value, str) else f"'{item.value}'" for item in values]
            values = f'[{", ".join(values)}]'
        return f'{alias}: {values}'

class Option(Domain):
    """ Alias for Domain({name: values}). """
    def __init__(self, name, values):
        super().__init__({name: values})

KV = Alias # is needed to load and transform old researches
