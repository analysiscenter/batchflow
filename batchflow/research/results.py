""" Research results. """

import os
import functools
from collections import OrderedDict
import glob
import dill
import multiprocess as mp
import pandas as pd
import numpy as np
from .utils import to_list

class ResearchResults:
    """ Class to collect, load and process research results.

    Parameters
    ----------
    name : str
        research name.
    dump_results : bool, optional
        does research dump results or not, by default True.
    """
    def __init__(self, name, dump_results=True, **kwargs):
        self.name = name
        self.dump_results = dump_results
        self.results = mp.Manager().dict()
        self.configs = mp.Manager().dict()

        self.kwargs = kwargs

    def put(self, experiment_id, results, config):
        self.results[experiment_id] = results
        self.configs[experiment_id] = config

    def load(self, **kwargs):
        """ Load (filtered if needed) results and configs if they was dumped. """
        self.kwargs = {**self.kwargs, **kwargs}
        if self.dump_results:
            self.load_configs()
            self.load_results(**self.kwargs)

    def load_configs(self):
        """ Load experiment configs. """
        for path in glob.glob(os.path.join(self.name, 'experiments', '*', 'config')):
            path = os.path.normpath(path)
            _experiment_id = path.split(os.sep)[-2]
            with open(path, 'rb') as f:
                self.configs[_experiment_id] = dill.load(f)

    def load_results(self, experiment_id=None, name=None, iterations=None,
                     config=None, alias=None, domain=None, **kwargs):
        """ Load and filter experiment results. """
        experiment_id, name, iterations = self.filter(experiment_id, name, iterations, config, alias, domain, **kwargs)
        results = dict()
        for path in glob.glob(os.path.join(self.name, 'experiments', '*', 'results', '*')):
            path = os.path.normpath(path)
            _experiment_id, _, _name = path.split(os.sep)[-3:]
            if experiment_id is None or _experiment_id in experiment_id:
                if name is None or _name in name:
                    if _experiment_id not in results:
                        results[_experiment_id] = OrderedDict()
                    experiment_results = results[_experiment_id]

                    if _name not in experiment_results:
                        experiment_results[_name] = OrderedDict()
                    name_results = experiment_results[_name]
                    new_values = self.load_iteration_files(path, iterations)
                    experiment_results[_name] = OrderedDict([*name_results.items(), *new_values.items()])
        self.results = mp.Manager().dict(**results)

    def filter(self, experiment_id=None, name=None, iteration=None, config=None, alias=None, domain=None, **kwargs):
        """ Filter results by specified parameters. """
        experiment_id = experiment_id if experiment_id is None else to_list(experiment_id)
        name = name if name is None else to_list(name)
        iteration = iteration if iteration is None else to_list(iteration)

        filtered_ids = self.filter_ids_by_configs(config, alias, domain, **kwargs)
        experiment_id = np.intersect1d(experiment_id, filtered_ids) if experiment_id is not None else filtered_ids

        return experiment_id, name, iteration

    @property
    def df(self):
        """ Create pandas.DataFrame from results. """
        return self.to_df()

    def to_df(self, pivot=False, include_config=True, use_alias=True, concat_config=False,
              remove_auxilary=True, **kwargs):
        """ Create pandas.DataFrame from filtered results. """
        if self.dump_results:
            self.load(**kwargs)

        kwargs = {**self.kwargs, **kwargs}
        experiment_ids, names, iterations = self.filter(**kwargs)

        df = []
        for experiment_id in experiment_ids:
            if experiment_id in self.results:
                experiment_df = []
                for name in (names or self.results[experiment_id]):
                    if name in self.results[experiment_id]:
                        _df = {
                            'id': experiment_id,
                            'iteration': self.results[experiment_id][name].keys()
                        }
                        if pivot:
                            _df[name] = self.results[experiment_id][name].values()
                        else:
                            _df['name'] = name
                            _df['value'] = self.results[experiment_id][name].values()
                        _df = pd.DataFrame(_df)
                        if iterations is not None:
                            _df = _df[_df.iteration.isin(iterations)]
                        experiment_df += [_df]
                if pivot and len(experiment_df) > 0:
                    experiment_df = [functools.reduce(functools.partial(pd.merge, on=['id', 'iteration']), experiment_df)]
                df += experiment_df
        res = pd.concat(df) if len(df) > 0 else pd.DataFrame()
        if include_config and len(res) > 0:
            res = pd.merge(res, self.configs_to_df(use_alias, concat_config, remove_auxilary), how='inner', on='id')
        return res

    def load_iteration_files(self, path, iterations):
        """ Load files for specified iterations. """
        filenames = glob.glob(os.path.join(path, '*'))
        if iterations is None:
            files_to_load = {int(os.path.basename(filename)): filename for filename in filenames}
        else:
            dumped_iteration = np.sort(np.array([int(os.path.basename(filename)) for filename in filenames]))
            files_to_load = dict()
            for iteration in iterations:
                _it = dumped_iteration[np.argwhere(dumped_iteration >= iteration)[0, 0]]
                files_to_load[_it] = os.path.join(path, str(_it))
        files_to_load = OrderedDict(sorted(files_to_load.items()))
        results = OrderedDict()
        for filename in files_to_load.values():
            with open(filename, 'rb') as f:
                values = dill.load(f)
                for iteration in values:
                    if iterations is None or iteration in iterations:
                        results[iteration] = values[iteration]
        return results

    def configs_to_df(self, use_alias=True, concat_config=False, remove_auxilary=True):
        """ Create pandas.DataFrame with configs. """
        df = []
        for experiment_id in self.configs:
            config = self.configs[experiment_id]
            if remove_auxilary:
                for key in ['repetition', 'device', 'updates']:
                    config.pop_config(key)
            if use_alias:
                if concat_config:
                    config = {'config': config.alias(as_string=concat_config)}
                else:
                    config = config.alias()
            else:
                config = config.config()
            df += [pd.DataFrame({'id': [experiment_id], **config})]
        return pd.concat(df)

    def filter_ids_by_configs(self, config=None, alias=None, domain=None, **kwargs):
        """ Filter configs.

        Parameters
        ----------
        repetition : int, optional
            index of the repetition to load, by default None
        experiment_id : str or list, optional
            experiment id to load, by default None
        configs : dict, optional
            specify keys and corresponding values to load results, by default None
        aliases : dict, optional
            the same as `configs` but specify aliases of parameters, by default None

        Returns
        -------
        list
            filtered list on configs
        """
        if sum([domain is not None, config is not None, alias is not None]) > 1:
            raise ValueError('Only one of `config`, `alias` and `domain` can be not None')
        filtered_ids = []
        if domain is not None:
            for _config in domain.iterator():
                filtered_ids += self.filter_ids_by_configs(config=_config.config())
            return filtered_ids

        if len(kwargs) > 0:
            if config is not None:
                config = {**config, **kwargs}
            elif alias is not None:
                alias = {**alias, **kwargs}
            else:
                config = kwargs

        if config is None and alias is None:
            return list(self.configs.keys())

        for experiment_id, supconfig in self.configs.items():
            if config is not None:
                _config = supconfig.config()
                if all(item in _config.items() for item in config.items()):
                    filtered_ids += [experiment_id]
            else:
                _config = supconfig.alias()
                if all(item in _config.items() for item in alias.items()):
                    filtered_ids += [experiment_id]
        return filtered_ids
