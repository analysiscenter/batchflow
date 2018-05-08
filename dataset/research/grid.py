#pylint:disable=too-few-public-methods

""" Options and configs. """

from itertools import product
from functools import reduce # Valid in Python 2.6+, required in Python 3
import operator
import collections

from ..config import Config

class KV:
    """ Class for value and alias. """
    def __init__(self, value, alias=None):
        """
        Parameters
        ----------
        value : obj
        alias : obj
            if None alias will be equal to value.
        """
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
            return value.__name__
        else:
            return str(value)

class Option:
    """ Class for single-parameter option. """
    def __init__(self, parameter, values):
        """
        Parameters
        ----------
        parameter : KV or obj
        values : list of KV or obj
        """
        self.parameter = KV(parameter)
        self.values = [KV(value) for value in values]

    def alias(self):
        """ Returns alias of the Option. """
        return {self.parameter.alias: [value.alias for value in self.values]}

    def option(self):
        """ Returns config. """
        return {self.parameter.value: [value.value for value in self.values]}

    @classmethod
    def product(cls, *args):
        """ Element-wise product of options. """
        lens = [len(item.values) for item in args]
        if len(set(lens)) != 1:
            raise ValueError('Options must be of the same length.')

        grid = Grid()
        for i in range(lens[0]):
            grid += reduce(operator.mul, [Option(item.parameter, [item.values[i]]) for item in args])
        return grid

    def __repr__(self):
        return 'Option(' + str(self.alias()) + ')'

    def __mul__(self, other):
        return Grid(self) * Grid(other)

    def __add__(self, other):
        return Grid(self) + Grid(other)

    def gen_configs(self, n_items=None):
        """ Returns Configs created from the option. """
        grid = Grid(self)
        return grid.gen_configs(n_items)

class ConfigAlias:
    """ Class for config. """
    def __init__(self, config):
        """
        Parameters
        ----------
        config : list of (key, value)
            keys and values are KV
        """
        self._config = config

    def alias(self, as_string=False, delim='-'):
        """ Returns alias. """
        config_alias = {item[0].alias: item[1].alias for item in self._config}
        if as_string is False:
            return config_alias
        else:
            config_alias = collections.OrderedDict(sorted(config_alias.items()))
            return delim.join([str(key)+'_'+str(value) for key, value in config_alias.items()])

    def config(self):
        """ Returns values. """
        return Config({item[0].value: item[1].value for item in self._config})

    def __repr__(self):
        return 'ConfigAlias(' + str(self.alias()) + ')'

class Grid:
    """ Class for grid of parameters. """
    def __init__(self, grid=None, **kwargs):
        """
        Parameters
        ----------
        grid: Option, Grid or list of lists of Options
        """
        if isinstance(grid, Option):
            self.grid = [[grid]]
        elif isinstance(grid, Grid):
            self.grid = grid.grid
        elif isinstance(grid, dict):
            self.grid = self._dict_to_grid(grid)
        else:
            self.grid = grid

        if len(kwargs) > 0:
            self.grid.append(self._dict_to_grid(kwargs))

    def _dict_to_grid(self, grid):
        _grid = []
        for key, value in grid.items():
            _grid.append(Option(key, value))
        return [_grid]

    def alias(self):
        """ Returns alias of Grid. """
        return [[option.alias() for option in options] for options in self.grid]

    def value(self):
        """ Returns config of Grid. """
        return [[option.option() for option in options] for options in self.grid]

    def description(self):
        """ Return description of used aliases.
        Returns
        -------
        dict
        """
        options = [option for grid_item in self.grid for option in grid_item]
        descr = dict()
        for option in options:
            values = {value.alias: value.value for value in option.values}
            if option.parameter.alias not in descr:
                descr[option.parameter.alias] = {'name': option.parameter.value, 'values': values}
            else:
                descr[option.parameter.alias]['values'].update(values)
        return descr

    def __len__(self):
        if self.grid is None:
            return 0
        else:
            return len(self.grid)

    def __mul__(self, other):
        if self.grid is None:
            return other
        elif isinstance(other, Grid):
            if other.grid is None:
                return self
            res = list(product(self.grid, other.grid))
            res = [item[0] + item[1] for item in res]
            return Grid(res)
        elif isinstance(other, Option):
            return self * Grid([[other]])

    def __add__(self, other):
        if self.grid is None:
            return other
        elif isinstance(other, Option):
            return self + Grid(other)
        elif other.grid is None:
            return self
        elif isinstance(other, Grid):
            if other.grid is None:
                return self
            else:
                return Grid(self.grid + other.grid)

    def __repr__(self):
        return 'Grid(' + str(self.alias()) + ')'

    def __getitem__(self, index):
        return Grid([self.grid[index]])

    def __eq__(self, other):
        return self.grid() == other.grid()

    def gen_configs(self, n_items=None):
        """ Generate Configs from grid. """
        for item in self.grid:
            keys = [option.parameter for option in item]
            values = [option.values for option in item]
            if n_items is None:
                for parameters in product(*values):
                    yield ConfigAlias(list(zip(keys, parameters)))
            else:
                res = []
                for parameters in product(*values):
                    if len(res) < n_items:
                        res.append(ConfigAlias(list(zip(keys, parameters))))
                    else:
                        yield res
                        res = [ConfigAlias(list(zip(keys, parameters)))]
                yield res
