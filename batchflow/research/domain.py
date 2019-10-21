""" Options and configs. """

from itertools import product, islice
from collections import OrderedDict
from copy import deepcopy
import numpy as np

from .. import Config, Sampler
from .named_expr import ResearchNamedExpression

class KV:
    """ Class for value and alias. Is used to create short and clear alias for some Python object
    that can be used as an element of research `Domain`.

    Parameters
    ----------
    value : hashable

    alias : object
        if None alias will be equal to `value`.
    """
    def __init__(self, value, alias=None):
        if isinstance(value, KV):
            self.value = value.value
            self.alias = value.alias
        else:
            self.value = value
            if alias is None:
                self.alias = self._get_name(value)
            else:
                self.alias = alias

    def __repr__(self):
        return 'KV(' + str(self.alias) + ': ' + str(self.value) + ')'

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
    parameter : KV or object

    values : list, tuple of KV or objects, np.ndarray or Sampler.
    """
    def __init__(self, parameter, values):
        self.parameter = KV(parameter)
        if isinstance(values, (list, tuple, np.ndarray)):
            self.values = [KV(value) for value in values]
        elif isinstance(self, Sampler):
            self.values = values
        else:
            raise TypeError('values must be array-like object or Sampler but {} were given'.format(type(values)))

    def __matmul__(self, other):
        if isinstance(self.values, Sampler) or isinstance(other.values, Sampler):
            domain = self * other
        elif len(self.values) == len(other.values):
            domain = Domain()
            for item in zip(self.values, other.values):
                domain += Option(self.parameter, [item[0]]) * Option(other.parameter, [item[1]])
        return domain

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Domain(self) * other
        return Domain(self) * Domain(other)

    def __rmul__(self, other):
        return self * other

    def __add__(self, other):
        return Domain(self) + Domain(other)

    def __repr__(self):
        if isinstance(self.values, (list, tuple, np.ndarray)):
            return 'Option({}, {})'.format(self.parameter.alias, [item.alias for item in self.values])
        else:
            return 'Option({}, {})'.format(self.parameter.alias, self.values)

    def sample(self, size=None):
        """ Return ConfigAlias objects created on the base of Sampler-option.

        Parameters
        ----------
        size : int or None
            the size of the sample

        Returns
        -------
            ConfigAlias if size is None, list of ConfigAlias objects else.
        """

        if not isinstance(self.values, Sampler):
            raise TypeError('values must be Sampler but {} was given'.format(type(self.values)))
        _size = size or 1
        res = [ConfigAlias([[self.parameter, self.values.sample(1)[0, 0]]]) for _ in range(_size)]
        if size is None:
            res = res[0]
        return res

    def items(self):
        """ Return all possible ConfigAliases which can be created from the option.

        Returns
        -------
            list of ConfigAlias objects.
        """
        if not isinstance(self.values, (list, tuple, np.ndarray)):
            raise TypeError('values must be array-like object but {} were given'.format(type(self.values)))
        return [ConfigAlias([[self.parameter, value]]) for value in self.values]

    def iterator(self):
        """ Produce ConfigAlias from the option.

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
    """ Class for config and alias which is represenation of config where all keys are str.

    Parameters
    ----------
    config : list of (key, value)
        keys are KV or hashable, value is KV or object.
    """
    def __init__(self, config=None):
        _config = []
        if config is not None:
            for key, value in config:
                _key = key[0] if isinstance(key[0], KV) else KV(key[0])
                _value = value[1] if isinstance(value[1], KV) else KV(value[1])
                _config.append((_key, _value))
        self._config = _config

    def alias(self, as_string=False, delim='-'):
        """ Returns alias.

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
        config_alias = {item[0].alias: item[1].alias for item in self._config}
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

    def __repr__(self):
        return 'ConfigAlias(' + str(self.alias()) + ')'

    def __add__(self, other):
        config = ConfigAlias()
        config._config = deepcopy(self._config) + deepcopy(other._config)
        return config

class Domain:
    """ Class for domain of parameters. `Domain` is a list of list of Options. Each list of Options
    will produce configs as an Cartesian multiplications of Option values.

    Parameters
    ----------
    domain : Option, Domain or list of lists of Options


    **Operations with Domains**

    #. sum by `+`: Concatenate list of Configs generated by Domains

    #. multiplication by `*`: Cartesian multiplications of Options in Domain.
       For example, if `domain1 = Option('a': [1, 2])` and
       `domain1 = Option('b': [3, 4])` then `domain1 * domain2` will have both options and generate 4 configs:
       `{a: 1, b: 3}`, `{a: 1, b: 4}`, `{a: 2, b: 3}`, `{a: 2, b: 4}`.
    """
    def __init__(self, domain=None, weights=None, **kwargs):
        if isinstance(domain, Option):
            self.cubes = [[domain]]
            self.weights = [np.nan]
        elif isinstance(domain, Domain):
            self.cubes = domain.cubes
            self.weights = domain.weights
        elif isinstance(domain, dict):
            self.cubes = self._dict_to_domain(domain)
            self.weights = [np.nan]
        elif isinstance(domain, list) and all([isinstance(item, list) for item in domain]):
            self.cubes = domain
            self.weights = [np.nan] * len(domain)
        else:
            raise ValueError('domain can be Option, Domain, dict or nested list but {} were given'.format(type(domain)))
        if len(kwargs) > 0:
            self.cubes = self._dict_to_domain(kwargs)
            self.weights = [np.nan]

        if weights is not None:
            self.weights = weights

        self.weights = np.array(self.weights)
        self.update_func = None
        self.update_each = None
        self.update_args = None
        self.update_kwargs = None

        self.iterator = None

        self.brute_force = []
        for cube in self.cubes:
            self.brute_force.append(not all([isinstance(option.values, Sampler) for option in cube]))

    def _dict_to_domain(self, domain):
        _domain = []
        for key, value in domain.items():
            _domain.append(Option(key, value))
        return [_domain]

    def __mul__(self, other):
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
        return 'Domain(' + str(self.cubes) + ')'

    def __getitem__(self, index):
        return Domain([self.cubes[index]])

    def __eq__(self, other):
        return self.cubes == other.cubes

    def __next__(self):
        if self.iterator is None:
            self.reset_iter()
        return next(self.iterator)

    def cube_iterator(self, cube):
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
        incl = np.cumsum(np.isnan(self.weights))
        excl = np.concatenate(([0], incl[:-1]))
        block_indices = incl + excl
        return [np.where(block_indices == i)[0] for i in set(block_indices)]

    def reset_iter(self, n_iters=None, n_reps=1, repeat_each=100):
        blocks = self._get_sampling_blocks()
        size = self._options_size()
        if n_iters is not None:
            repeat_each = n_iters
        elif size is not None:
            repeat_each = size
        def _iterator():
            while True:
                for block in blocks:
                    weights = self.weights[block]
                    weights[np.isnan(weights)] = 1
                    iterators = [self.cube_iterator(cube) for cube in np.array(self.cubes)[block]]
                    while len(iterators) > 0:
                        index = np.random.choice(len(block), p=weights / weights.sum())
                        try:
                            yield next(iterators[index])
                        except StopIteration:
                            del iterators[index]
                            weights = np.delete(weights, index)
                            block = np.delete(block, index)

        def _iterator_with_repetitions():
            iterator = _iterator()
            if n_reps == 1:
                i = 0
                while n_iters is None or i < n_iters:
                    yield next(iterator)
                    i += 1
            else:
                i = 0
                while n_iters is None or i < n_iters:
                    samples = list(islice(iterator, repeat_each))
                    for repetition in range(n_reps):
                        for sample in samples:
                            yield sample + ConfigAlias([('repetition', repetition)])
                    i += repeat_each
        self.n_iters = n_iters
        self.n_reps = n_reps
        self.iterator = _iterator_with_repetitions()

    def _options_size(self):
        size = 0
        for cube in self.cubes:
            lengthes = [len(option.values) for option in cube if isinstance(option.values, (list, tuple, np.ndarray))]
            if len(lengthes) == 0:
                return None
            else:
                size += np.product(lengthes)
        return size
    
    @property
    def size(self):
        if self.n_iters is not None:
            return self.n_reps * self.n_iters
        else:
            return None

    def update_domain(self, path):
        if self.update_func is None:
            return None
        args = ResearchNamedExpression.eval_expr(self.args, path=path)
        kwargs = ResearchNamedExpression.eval_expr(self.kwargs, path=path)
        return self.update_func(*args, **kwargs)

    def update_config(self, *args, **kwargs):
        _ = args, kwargs
        return None
