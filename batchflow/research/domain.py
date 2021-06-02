""" Options and configs. """

from batchflow.research.utils import must_execute
from itertools import product, islice
from collections import OrderedDict
from copy import copy, deepcopy
import numpy as np

from .. import Config, Sampler
from ..named_expr import eval_expr

class Alias:
    """ Class for value and its alias. Is used to create short and clear alias for some Python object
    that can be used as an element of `Domain`.

    Parameters
    ----------
    value : hashable

    alias : object
        if None, alias will be equal to `value.__name__` or to `str(value)`.
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
        if hasattr(value, '__name__'):
            name = value.__name__
        else:
            name = str(value)
        return name

class Option:
    """ Class for single-parameter option. There is an algebra of options (see :class:`~.Domain` operations)
    Result is a `Domain`.

    Parameters
    ----------
    parameter : Alias or str
        parameter name
    values : list, tuple, numpy.ndarray or Sampler
        parameter values. Elements of list or tuple can be Aliases.
    """
    def __init__(self, parameter, values):
        self.parameter = Alias(parameter)
        if isinstance(values, (list, tuple, np.ndarray)):
            self.values = [Alias(value) for value in values]
        elif isinstance(values, Sampler):
            self.values = values
        else:
            raise TypeError('`values` must be array-like object or Sampler but {} were given'.format(type(values)))

    def __matmul__(self, other):
        return Domain(self) @ Domain(other)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Domain(self) * other
        return Domain(self) * Domain(other)

    def __rmul__(self, other):
        return self * other

    def __add__(self, other):
        return Domain(self) + Domain(other)

    def __repr__(self):
        alias = self.parameter.alias
        values = self.values

        if isinstance(self.values, (list, tuple, np.ndarray)):
            values = [item.alias if not isinstance(item.value, str) else f"'{item.value}'" for item in values]
            values = f'[{",".join(values)}]'
        return f'Option({alias}: {values})'

    def sample(self, size=None):
        """ Return `ConfigAlias` objects created on the base of Sampler-option.

        Parameters
        ----------
        size : int or None
            the size of the sample

        Returns
        -------
            ConfigAlias (if size is None) or list of ConfigAlias objects (otherwise).
        """

        if not isinstance(self.values, Sampler):
            raise TypeError('`values` must be Sampler but {} was given'.format(type(self.values)))
        res = [ConfigAlias([[self.parameter, self.values.sample(1)[0, 0]]]) for _ in range(size or 1)]
        if size is None:
            res = res[0]
        return res

    def items(self):
        """ Return all possible `ConfigAlias` instances which can be created from the option.

        Returns
        -------
            list of `ConfigAlias` objects.
        """
        if not isinstance(self.values, (list, tuple, np.ndarray)):
            raise TypeError('`values` must be array-like object but {} were given'.format(type(self.values)))
        return [ConfigAlias([[self.parameter, value]]) for value in self.values]

    def iterator(self):
        """ Produce `ConfigAlias` from the option.

        Returns
        -------
            generator.
        """
        if isinstance(self.values, Sampler):
            while True:
                yield ConfigAlias([[self.parameter, self.values.sample(1)[0, 0]]])
        else:
            for value in self.values:
                yield ConfigAlias([[self.parameter, value]])


class ConfigAlias:
    """ Class for config and its aliased version where all keys and values are `str`.

    Parameters
    ----------
    config : list of (key, value)
        each key is `Alias` or str, value is `Alias` or object.
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
        as_string : bool
            if True, return string representation of ConfigAlias. Different items will be
            separated by `delim`, key and value for each pair will be separated by '_'.
        delim : str
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
        """ Returns values of ConfigAlias.

        Returns
        -------
            Config
        """
        return Config({item[0].value: item[1].value for item in self._config})

    def pop_config(self, key):
        """ Pop item from ConfigAlias by config value.

        Returns
        -------
            ConfigAlias or None
        """
        res = [item for item in self._config if item[0].value == key]
        self._config = [item for item in self._config if item[0].value != key]
        if len(res) == 1:
            return ConfigAlias(res)
        return None

    def pop_alias(self, key):
        """ Pop item from ConfigAlias by alias value.

        Returns
        -------
            ConfigAlias or None
        """
        res = [item for item in self._config if item[0].alias == key]
        self._config = [item for item in self._config if item[0].alias != key]
        if len(res) == 1:
            return ConfigAlias(res)
        return None

    def __getitem__(self, key):
        return self.config()[key]

    def __repr__(self):
        from pprint import pformat
        return pformat(self.alias().config)

    def __add__(self, other):
        config = ConfigAlias()
        config._config = deepcopy(self._config) + deepcopy(other._config)
        return config

    def keys(self):
        return self.config().keys()

class Domain:
    """ Class for domain of parameters which produce ConfigAlias objects. `Domain` is a result
    of algebraic operations on options. The inner structure of `Domain` is a list of list of Options.
    The inner lists are called cubes because each list of parameters corresponds to some cube in
    multidimensional space of parameters.
    Each cube will produce configs as an Cartesian multiplications of array-Option values and other
    Sampler-Option will be sampled for each combination of parameters independently.

    Parameters
    ----------
    domain : Option, Domain, dict or list of lists of Options


    **Operations with Domains**

    #. sum by `+`: Concatenate lists of cubes

    #. multiplication by `*`: Cartesian multiplications of Options in Domain.
       For example, if `domain1 = Option('a': [1, 2])`, `domain2 = Option('b': [3, 4])` and
       `domain3 = Option('c': bf.Sampler('n'))` then `domain1 * domain2 * domain3` will have
       all options and generate 4 configs: `{'a': 1, 'b': 3, 'c': xi_1}`, `{'a': 1, 'b': 4, 'c': xi_2}`,
       `{'a': 2, 'b': 3, 'c': xi_3}`, `{'a': 2, 'b': 4, 'c': xi_4}`
       where xi_i are independent samples from normal distribution.
    #. multiplication by @: element-wise multiplication of array-like Options.
       For example, if `domain1 = Option('a': [1, 2])` and `domain2 = Option('b': [3, 4])` then
       `domain1 @ domain2` will have two configs:
       `{'a': 1, `b`: 3}`, `{'a': 2, `b`: 4}`.
    #. multiplication with weights: can be used to sample configs from sum of Options
        For example, `0.3 * Option('p1', NS('n', loc=-10)) + 0.2 * Option('p2',  NS('u'))
        + 0.5 * Option('p3',  NS('n', loc=10))` will return `{'p1': -10.3059}, {'p3': 8.9959},
        {'p3': 9.1302}, {'p3': 10.2611}, {'p1': -7.9388}, {'p2': 0.5455}, {'p1': -9.2497},
        {'p3': 9.9769}, {'p2': 0.3510}, {'p3': 8.8519}` (depends on seed).
        See more in tutorials (`<../../examples/tutorials/research/04_advance_usage_of_domain.ipynb>_`).
    """
    def __init__(self, domain=None, weights=None, **kwargs):
        if isinstance(domain, Option):
            self.cubes = [[domain]]
            self.weights = [np.nan]
        elif isinstance(domain, Domain):
            self.cubes = copy(domain.cubes)
            self.weights = copy(domain.weights)
        elif isinstance(domain, dict):
            self.cubes = self._dict_to_domain(domain)
            self.weights = [np.nan]
        elif isinstance(domain, list) and all(isinstance(item, list) for item in domain):
            self.cubes = domain
            self.weights = [np.nan] * len(domain)
        elif domain is None:
            self.cubes = []
            self.weights = []
        else:
            raise ValueError('domain can be Option, Domain, dict or nested list but {} were given'.format(type(domain)))
        if len(kwargs) > 0:
            self.cubes = self._dict_to_domain(kwargs)
            self.weights = [np.nan]

        if weights is not None:
            self.weights = weights

        self.weights = np.array(self.weights) # weights for each cube to sample values from it
        self.updates = []
        self.n_produced = 0

        self._iterator = None
        self.n_items = None
        self.n_reps = 1
        self.repeat_each = None
        self.n_updates = 0

        self.brute_force = []
        for cube in self.cubes:
            self.brute_force.append(not all(isinstance(option.values, Sampler) for option in cube))

    def _dict_to_domain(self, domain):
        _domain = []
        for key, value in domain.items():
            _domain.append(Option(key, value))
        return [_domain]

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
            result = Domain(res, weights=weights)
        elif isinstance(other, Option):
            result = self * Domain(other)
        else:
            raise TypeError('Arguments must be numeric, Domains or Options')
        return result

    def __matmul__(self, other):
        if isinstance(other, Option):
            return self @ Domain(other)

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
                return Domain(domain=cubes, weights=weights)
        raise ValueError("The numbers of domain cubes must conincide.")

    def __rmul__(self, other):
        return self * other

    def __add__(self, other):
        if self.cubes is None:
            result = other
        elif isinstance(other, Option):
            result = self + Domain(other)
        elif other.cubes is None:
            result = self
        elif isinstance(other, Domain):
            result = Domain(self.cubes + other.cubes, weights=np.concatenate((self.weights, other.weights)))
        return result

    def __repr__(self):
        repr = ''
        cubes_reprs = []
        spacing = 4 * ' '

        for cube in self.cubes:
            cubes_reprs += [' * '.join([str(option) for option in cube])]
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

    def __getitem__(self, index):
        return Domain([self.cubes[index]])

    def __eq__(self, other):
        return self.cubes == other.cubes

    def __next__(self):
        return next(self.iterator())

    @property
    def size(self):
        """ Return the number of configs that will be produces from domain. """
        if self.n_items is not None:
            return self.n_reps * self.n_items
        return None

    def __len__(self):
        """ Return the number of configs that will be produced from domain without repetitions. """
        cube_sizes = [
            np.prod([len(op.values) for op in cube if isinstance(op.values, (list, tuple, np.ndarray))], dtype='int')
            for cube in self.cubes
        ]
        return max(0, sum(cube_sizes))

    def reset_iter(self):
        self._iterator = None

    def set_iter_params(self, n_items=None, n_reps=1, repeat_each=None, produced=0): #TODO: produce infinite sequencies for samplers
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
        """
        n_configs = len(self)
        self.n_items = n_items or n_configs
        self.n_reps = n_reps
        if self.n_items is not None:
            self.repeat_each = repeat_each or self.n_items
        elif n_configs is not None:
            self.repeat_each = repeat_each or n_configs
        else:
            self.repeat_each = repeat_each or 100
        self.n_produced = produced

        self.reset_iter()

    def create_iter(self):
        blocks = self._get_sampling_blocks()

        def _iterator():
            while True:
                for block in blocks:
                    weights = self.weights[block]
                    weights[np.isnan(weights)] = 1
                    iterators = [self._cube_iterator(cube) for cube in np.array(self.cubes)[block]]
                    while len(iterators) > 0:
                        index = np.random.choice(len(block), p=weights/weights.sum())
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
                additional = ConfigAlias([('repetition', 0)]) + ConfigAlias([('updates', self.n_updates)])
                while self.n_items is None or i < self.n_items:
                    yield next(iterator) + additional # pylint: disable=stop-iteration-return
                    i += 1
            else:
                i = 0
                while self.n_items is None or i < self.n_items:
                    samples = list(islice(iterator, int(self.repeat_each)))
                    for repetition in range(self.n_reps):
                        additional = ConfigAlias({'repetition': repetition}) + ConfigAlias({'updates': self.n_updates})
                        for sample in samples:
                            yield sample + additional
                    i += self.repeat_each
        self._iterator = _iterator_with_repetitions()

    def iterator(self):
        """ Get domain iterator. """
        if self._iterator is None:
            self.set_iter_params(self.n_items, self.n_reps, self.repeat_each, self.n_produced)
            self.create_iter()
        return self._iterator

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
                new_domain = Domain(update['function'](**kwargs))
                new_domain.updates = self.updates
                new_domain.n_updates = self.n_updates + 1
                new_domain.set_iter_params(produced=generated, **update['iter_kwargs'])
                return new_domain
        return None

    def _is_array_option(self):
        """ Return True if domain consists of only on array-like option. """
        if len(self.cubes) == 1:
            if len(self.cubes[0]) == 1:
                if isinstance(self.cubes[0][0].values, (list, tuple, np.ndarray)):
                    return True
        return False

    def _is_scalar_product(self):
        """ Return True if domain is a result of matmul. It means that each cube has
        an only one array-like option of length 1.
        """
        for cube in self.cubes:
            samplers = [option for option in cube if isinstance(option.values, Sampler)]
            if len(samplers) > 0:
                return False
            if any(len(item.values) != 1 for item in cube):
                return False
        return True

    def _to_scalar_product(self):
        """ Transform domain to the matmul format (see :meth:`~.Domain._is_scalar_product`)"""
        if self._is_array_option():
            option = self.cubes[0][0]
            cubes = [[Option(option.parameter, [value])] for value in option.values]
            weights = np.concatenate([[self.weights[0]] * len(cubes)])
            return Domain(cubes, weights)
        if self._is_scalar_product():
            return Domain(self)
        raise ValueError("Domain cannot be represented as scalar product.")

    def _cube_iterator(self, cube):
        """ Return iterator from the cube. All array-like options will be transformed
        to Cartesian product and all sampler-like options will produce independent samples
        for each condig. """
        arrays = [option for option in cube if isinstance(option.values, (list, tuple, np.ndarray))]
        samplers = [option for option in cube if isinstance(option.values, Sampler)]

        if len(arrays) > 0:
            for combination in list(product(*[option.items() for option in arrays])):
                res = []
                for option in samplers:
                    res.append(option.sample())
                res.extend(combination)
                yield sum(res, ConfigAlias())
        else:
            iterators = [option.iterator() for option in cube]
            while True:
                try:
                    yield sum([next(iterator) for iterator in iterators], ConfigAlias())
                except StopIteration:
                    break

    def _get_sampling_blocks(self):
        """ Return blocks for sampling on the basis of weights. """
        incl = np.cumsum(np.isnan(self.weights))
        excl = np.concatenate(([0], incl[:-1]))
        block_indices = incl + excl
        return [np.where(block_indices == i)[0] for i in set(block_indices)]
