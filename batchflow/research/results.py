""" Research results. """

import os
import functools
from collections import OrderedDict
import glob
import multiprocess as mp
import pandas as pd
import numpy as np

from ..utils import to_list
from .utils import deserialize

class ResearchResults:
    """ Class to collect, load and process research results.

    Parameters
    ----------
    name : str
        research name.
    dump_results : bool, optional
        does research dump results or not, by default True.
    kwargs :
        filtering kwargs for load.
    """
    def __init__(self, name, dump_results=True, **kwargs):
        self.name = name
        self.dump_results = dump_results
        self._manager = mp.Manager()
        self.results = self._manager.dict()
        self.configs = self._manager.dict()
        self.artifacts = dict()
        self.kwargs = kwargs

        if dump_results and os.path.exists(name):
            self.load()

    def put(self, experiment_id, results, config):
        self.results[experiment_id] = results
        self.configs[experiment_id] = config

    def load(self, **kwargs):
        """ Load (filtered if needed) results, configs and artifacts paths if they was dumped. """
        kwargs = {**self.kwargs, **kwargs}
        if self.dump_results:
            self.load_configs()
            self.load_results(**kwargs)
            self.load_artifacts(**kwargs)
        self.results = dict(self.results)
        self.configs = dict(self.configs)
        self.close_manager()

    def load_configs(self):
        """ Load all experiment configs. """
        for path in glob.glob(os.path.join(self.name, 'experiments', '*', 'config.dill')):
            path = os.path.normpath(path)
            _experiment_id = path.split(os.sep)[-2]
            with open(path, 'rb') as f:
                self.configs[_experiment_id] = deserialize(f)

    def load_results(self, experiment_id=None, name=None, iterations=None,
                     config=None, alias=None, domain=None, **kwargs):
        """ Load and filter experiment results.

        Parameters
        ----------
        experiment_id : str or list, optional
            exepriments to load, by default None.
        name : str or list, optional
            keys of results to load, by default None.
        iterations : int or list, optional
            iterations to load, by default None.
        config : Config, optional
            config with parameters values to load, by default None.
        alias : Config, optional
            the same as config but with aliased values, by default None.
        domain : Domain, optional
            domain with parameters values to load, by default None.
        kwargs : dict
            is used as `config`. If `config` is not defined but `alias` is, then will be concated to `alias`.
        """
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
        self.results = results

    def load_artifacts(self, experiment_id=None, name=None, config=None, alias=None, domain=None, **kwargs):
        """ Load and filter experiment artifacts (all files/folders in experiment folder except standart
        'results', 'config.dill', 'config.json', 'experiment.log').

        Parameters
        ----------
        experiment_id : str or list, optional
            exepriments to load, by default None
        name : str or list, optional
            names of artifacts to load into artifacts list, by default None
        config : Config, optional
            config with parameters values to load, by default None
        alias : Config, optional
            the same as config but with aliased values, by default None
        domain : Domain, optional
            domain with parameters values to load, by default None
        kwargs : dict
            is used as `config`. If `config` is not defined but `alias` is, then will be concated to `alias`.
        """
        self.artifacts = dict()
        names = to_list('*' if name is None else name)
        experiment_id, _, _ = self.filter(experiment_id, None, None, config, alias, domain, **kwargs)
        for _name in names:
            for path in glob.glob(os.path.join(self.name, 'experiments', '*', _name)):
                if os.path.basename(path) not in ['results', 'config.dill', 'config.json', 'experiment.log']:
                    path = os.path.normpath(path)
                    _experiment_id, _name = path.split(os.sep)[-2:]
                    if experiment_id is None or _experiment_id in experiment_id:
                        if _experiment_id not in self.artifacts:
                            self.artifacts[_experiment_id] = []
                        self.artifacts[_experiment_id] += [
                            {'artifact_name': _name,
                            'full_path': path,
                            'relative_path': os.path.join(*path.split(os.sep)[-3:])
                            }
                        ]

    def filter(self, experiment_id=None, name=None, iterations=None, config=None, alias=None, domain=None, **kwargs):
        """ Filter experiment_id by specified parameters and convert `name`, `iterations` to lists.

        Parameters
        ----------
        experiment_id : str or list, optional
            exepriments to load, by default None
        name : str or list, optional
            keys of results to load, by default None
        iterations : int or list, optional
            iterations to load, by default None
        config : Config, optional
            config with parameters values to load, by default None
        alias : Config, optional
            the same as config but with aliased values, by default None
        domain : Domain, optional
            domain with parameters values to load, by default None
        kwargs : dict
            is used as `config`. If `config` is not defined but `alias` is, then will be concated to `alias`.
        """
        experiment_id = experiment_id if experiment_id is None else to_list(experiment_id)
        name = name if name is None else to_list(name)
        iterations = iterations if iterations is None else to_list(iterations)

        filtered_ids = self.filter_ids_by_configs(config, alias, domain, **kwargs)
        experiment_id = np.intersect1d(experiment_id, filtered_ids) if experiment_id is not None else filtered_ids

        return experiment_id, name, iterations

    @property
    def df(self):
        """ Create pandas.DataFrame from results. """
        return self.to_df()

    def to_df(self, pivot=True, include_config=True, use_alias=False, concat_config=False,
              remove_auxilary=True, drop_columns=True, **kwargs):
        """ Create pandas.DataFrame from filtered results.

        Parameters
        ----------
        pivot : bool, optional
            if True, two columns will be created: `name` (for results variable) and `value`. If False, for each
            variable separate column will be created. By default True
        include_config : bool, optional
            include config into dataframe or not, by default True
        use_alias : bool, optional
            use alias of config values or not, by default True
        concat_config : bool, optional
            create one column for config (it will be concated) or create columns for each config parameter,
            by default False
        remove_auxilary : bool, optional
            remove columns 'repetition', 'device', 'updates' or not, by default True
        drop_columns : bool, optional
            remove or not separate columns for config parametrs when `concat_config=True`.
        kwargs : dict
            kwargs for :meth:`~.filter`.

        Returns
        -------
        pandas.DataFrame
        """
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
                    experiment_df = [
                        functools.reduce(
                            functools.partial(pd.merge, on=['id', 'iteration'], how='outer'),
                            experiment_df
                        )
                    ]
                df += experiment_df
        res = pd.concat(df) if len(df) > 0 else pd.DataFrame()
        if include_config and len(res) > 0:
            left = self.configs_to_df(use_alias, concat_config, remove_auxilary, drop_columns)
            res = pd.merge(left, res, how='inner', on='id')
        return res

    def load_iteration_files(self, path, iterations):
        """ Load files for specified iterations from specified path. """
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
                values = deserialize(f)
                for iteration in values:
                    if iterations is None or iteration in iterations:
                        results[iteration] = values[iteration]
        return results

    def configs_to_df(self, use_alias=True, concat_config=False, remove_auxilary=True, drop_columns=True):
        """ Create pandas.DataFrame with configs.

        Parameters
        ----------
        use_alias : bool, optional
            use alias of config values or not, by default True
        concat_config : bool, optional
            create one column for config (it will be concated) or create columns for each config parameter,
            by default False
        remove_auxilary : bool, optional
            remove columns 'repetition', 'device', 'updates' or not, by default True
        drop_columns : bool, optional
            remove or not separate columns for config parametrs when `concat_config=True`.

        Returns
        -------
        pandas.DataFrame
        """
        df = []
        for experiment_id in self.configs:
            config = self.configs[experiment_id]
            if remove_auxilary:
                for key in ['repetition', 'device', 'updates']:
                    config.pop_config(key)
            if concat_config:
                popped = config.pop_config(['repetition', 'device', 'updates'])
                if popped is None:
                    popped = {}
                else:
                    popped = popped.alias() if use_alias else popped.config()
                _config = {'config': config.alias(as_string=concat_config), **popped}
            else:
                _config = {}

            if not concat_config or not drop_columns:
                if use_alias:
                    _config = {**_config, **config.alias()}
                else:
                    _config = {**_config, **config.config()}

            df += [pd.DataFrame({'id': [experiment_id], **{key: [val] for key, val in _config.items()}})]
        return pd.concat(df)

    def artifacts_to_df(self, include_config=True, use_alias=False, concat_config=False,
                        remove_auxilary=True, drop_columns=True, **kwargs):
        """ Create pandas.DataFrame with experiment artifacts (all in experiment folder except standart
        'results', 'config.dill', 'config.json', 'experiment.log').

        Parameters
        ----------
        use_alias : bool, optional
            use alias of config values or not, by default True
        concat_config : bool, optional
            create one column for config (it will be concated) or create columns for each config parameter,
            by default False
        remove_auxilary : bool, optional
            remove columns 'repetition', 'device', 'updates' or not, by default True
        drop_columns : bool, optional
            remove or not separate columns for config parametrs when `concat_config=True`.
        kwargs : dict, optional
            filtering kwargs for :meth:`~.load_artifacts`.

        Returns
        -------
        pandas.DataFrame
            dataframe with name of the id of the experiment, artifact full path (with path to research folder)
            and relative path (inner path in research folder). Also can include experiment config.
        """
        if self.dump_results:
            self.load_configs()
            self.load_artifacts(**kwargs)
            df = []
            for experiment_id in self.artifacts:
                artifacts = self.artifacts[experiment_id]
                df += [pd.DataFrame({'id': [experiment_id], **artifact}) for artifact in artifacts]
            if len(df) > 0:
                df = pd.concat(df)
                if include_config:
                    df = pd.merge(self.configs_to_df(use_alias, concat_config, remove_auxilary, drop_columns),
                                df, how='inner', on='id')
                return df
            return pd.DataFrame({})
        raise ValueError("Research without dump can't have artifacts.")

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
            domain.reset_iter()
            for _config in domain.iterator:
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

    def close_manager(self):
        """ Close manager. """
        self.results = dict(self.results)
        self.configs = dict(self.configs)
        self._manager.shutdown()
