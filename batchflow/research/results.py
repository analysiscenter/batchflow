""" Class Results """

import os
import glob
from collections import OrderedDict
import dill
import pandas as pd
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
        files = {file: int(file.split('_')[-1]) for file in files}
        files = OrderedDict(sorted(files.items(), key=lambda x: x[1]))
        result = []
        start = 0
        for name, end in files.items():
            intersection = pd.np.intersect1d(iterations, pd.np.arange(start, end))
            if len(intersection) > 0:
                result.append((name, intersection))
            start = end
        return OrderedDict(result)

    def _slice_file(self, dumped_file, iterations_to_load, variables):
        iterations = dumped_file['iteration']
        if len(iterations) > 0:
            elements_to_load = pd.np.array([pd.np.isin(it, iterations_to_load) for it in iterations])
            res = OrderedDict()
            for variable in ['iteration', *variables]:
                if variable in dumped_file:
                    res[variable] = pd.np.array(dumped_file[variable])[elements_to_load]
        else:
            res = None
        return res

    def _concat(self, results, variables):
        res = {key: [] for key in [*variables, 'iteration']}
        for chunk in results:
            if chunk is not None:
                for key, values in res.items():
                    if key in chunk:
                        values.extend(chunk[key])
        return res

    def _fix_length(self, chunk):
        max_len = max([len(value) for value in chunk.values()])
        for value in chunk.values():
            if len(value) < max_len:
                value.extend([pd.np.nan] * (max_len - len(value)))

    def _filter_configs(self, config=None, alias=None):
        result = None
        if config is None and alias is None:
            raise ValueError('At least one of parameters config and alias must be not None')
        if config is not None:
            result = self.configs.subset(config, by_alias=False)
        else:
            result = self.configs.subset(alias, by_alias=True)
        return result


    def load(self, names=None, repetitions=None, variables=None, configs=None,
             iterations=None, aliases=None, use_alias=False):
        """ Load results as pandas.DataFrame.

        Parameters
        ----------
        names : str, list or None
            names of units (pipleines and functions) to load
        repetitions : int, list or None
            numbers of repetitions to load
        variables : str, list or None
            names of variables to load
        configs, aliases : dict, Config, Option, Grid or None
            configs to load
        iterations : int, list or None
            iterations to load
        use_alias : bool
            if True, the resulting DataFrame will have one column with alias, else it will
            have column for each option in grid
        """
        self.configs = self.research.grid_config
        if configs is None and aliases is None:
            self.configs = list(self.configs.gen_configs())
        elif configs is not None:
            self.configs = self._filter_configs(config=configs)
        else:
            self.configs = self._filter_configs(alias=aliases)

        if names is None:
            names = list(self.research.executables.keys())

        if repetitions is None:
            repetitions = list(range(self.research.n_reps))

        if variables is None:
            variables = [variable for unit in self.research.executables.values() for variable in unit.variables]

        if iterations is None:
            iterations = list(range(self.research.n_iters))

        self.names = self._get_list(names)
        self.repetitions = self._get_list(repetitions)
        self.variables = self._get_list(variables)
        self.iterations = self._get_list(iterations)

        data_frame = []

        for config_alias in self.configs:
            alias = config_alias.alias(as_string=False)
            alias_str = config_alias.alias(as_string=True)
            for repetition in self.repetitions:
                for unit in self.names:
                    path = os.path.join(self.path, 'results', alias_str, str(repetition))
                    files = glob.glob(os.path.join(glob.escape(path), unit + '_[0-9]*'))
                    files = self._sort_files(files, self.iterations)
                    if len(files) != 0:
                        res = []
                        for filename, iterations_to_load in files.items():
                            with open(filename, 'rb') as file:
                                res.append(self._slice_file(dill.load(file), iterations_to_load, self.variables))
                        res = self._concat(res, self.variables)
                        self._fix_length(res)
                        if use_alias:
                            data_frame.append(
                                pd.DataFrame({
                                    'config': alias_str,
                                    'repetition': repetition,
                                    'name': unit,
                                    **res
                                })
                            )
                        else:
                            data_frame.append(
                                pd.DataFrame({
                                    **alias,
                                    'repetition': repetition,
                                    'name': unit,
                                    **res
                                })
                                )
        return pd.concat(data_frame, ignore_index=True) if len(data_frame) > 0 else pd.DataFrame(None)
