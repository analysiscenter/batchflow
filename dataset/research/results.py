""" Class Results """

import os
import glob
import dill
import pandas as pd
from collections import OrderedDict
from .. import Config
from .research import Research

class Results():
    """ Class for dealing with results of research

    Parameters
    ----------
    path : str
        path to root folder of research
    research : Research
        instance of Research 
    """
    def __init__(self, path=None, research=None):
        if path is None and research is None:
            raise ValueError('At least one of parameters path and research must be not None')
        if path is None:
            self.research = research
            self.path = research.name
        else:
            self.research = Research().load(path)
            self.path = path
        self.results = dict()

    def _get_list(self, value):
        if not isinstance(value, list):
            value = [value]
        return value

    def _sort_files(self, files, iterations):
        files = OrderedDict({file: int(file.split('_')[-1]) for file in files})
        return OrderedDict(sorted(files.items(), key=lambda x: x[1])).keys()

    def _concat(self, results, variables):
        res = {key: [] for key in [*variables, 'iteration']}
        for chunk in results:
            for key, values in res.items():
                if key in chunk:
                    values.extend(chunk[key])
        return res

    def _fix_length(self, chunk):
        max_len = max([len(value) for value in chunk.values()])
        for key, value in chunk.items():
            if len(value) < max_len:
                value.extend([pd.np.nan] * (max_len - len(value)))

    def load(self, units=None, repetitions=None, variables=None, use_alias=False):
        """ Load results as pandas.DataFrame.

        Parameters
        ----------
        units : str, list or None
            names of pipleines and functions to load
        repetitions : int, list or None
            numbers of repetitions to load
        variables : str, list or None
            names of variables to load
        use_alias : bool
            if True, the resulting DataFrame will have one column with alias, else it will
            have column for each option in grid
        """
        self.configs = [config for config in self.research.grid_config.gen_configs()]

        if units is None:
            units = list(self.research.executables.keys())
        
        if repetitions is None:
            repetitions = list(range(self.research.n_reps))

        if variables is None:
            variables = [variable for unit in self.research.executables.values() for variable in unit.variables]

        self.units = self._get_list(units)
        self.repetitions = self._get_list(repetitions)
        self.variables = self._get_list(variables)

        self.iterations = list(range(self.research.n_iters))

        df = []

        for config_alias in self.configs:
            config = config_alias.config()
            alias = config_alias.alias(as_string=False)
            alias_str = config_alias.alias(as_string=True)
            for repetition in self.repetitions:
                for unit in self.units:
                    path = os.path.join(self.path, 'results', alias_str, str(repetition))
                    files = glob.glob(os.path.join(glob.escape(path), unit + '_*'))
                    files = self._sort_files(files, self.iterations)
                    if len(files) != 0:
                        res = []
                        for filename in files:
                            with open(filename, 'rb') as file:
                                res.append(dill.load(file))
                        res = self._concat(res, self.variables)
                        self._fix_length(res)
                        if use_alias:
                            df.append(pd.DataFrame({
                                'config': alias_str,
                                'repetition': repetition,
                                'unit': unit, 
                                **res
                            }))
                        else:
                            df.append(pd.DataFrame({**alias, 'repetition': repetition, 'unit': unit, **res}))
        return pd.concat(df)
