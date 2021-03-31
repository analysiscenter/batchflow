""" Research results class """

import os
from collections import OrderedDict
import glob
import json
import dill
import numpy as np
import pandas as pd

from .utils import to_list

class Results:
    """ Class for dealing with results of research

    Parameters
    ----------
    path : str
        path to root folder of research

    **How to perform slicing**
        Method `load` with default parameters will create pandas.DataFrame with all dumped
        parameters. To specify subset of results one can define names of pipelines/functions,
        produced variables/outputs of them, iterations and configs. For example,
        we have the following research:

        ```
        domain = Option('layout', ['cna', 'can', 'acn']) * Option('model', [VGG7, VGG16])

        research = (Research()
        .add_pipeline(train_ppl, variables='loss', name='train')
        .add_pipeline(test_ppl, name='test', execute=100, run=True, import_from='train')
        .add_callable(accuracy, returns='accuracy', name='test_accuracy',
                    execute=100, pipeline='test')
        .add_domain(domain))

        research.run(n_iters=10000)
        ```
        The code
        ```
        Results(research=research).load(iterations=np.arange(5000, 10000),
                                        variables='accuracy', names='test_accuracy',
                                        configs=Option('layout', ['cna', 'can']))
        ```
        will load output of ``accuracy`` function for configs
        that contain layout 'cna' or 'can' for iterations starting with 5000.
        The resulting dataframe will have columns 'iteration', 'name',
        'accuracy', 'layout', 'model'. One can get the same in the follwing way:
        ```
        results = Results(research=research).load()
        results = results[(results.iterations >= 5000) &
                            (results.name == 'test_accuracy') & results.layout.isin(['cna', 'can'])]
        ```
    """
    def __init__(self, path):
        self.path = path
        self.description = self._get_description()
        self.configs = self._load_configs()
        self.names = list(self.description['executables'].keys())
        self.variables = [var for unit in self.description['executables'].values() for var in unit['variables']]

    @property
    def df(self):
        return self._load_df()

    @property
    def artifacts(self):
        return self._load_artifactes()

    def load_df(self, *args, **kwargs):
        """ Load dataframe with results.

        Parameters
        ----------
        names : list, optional
            names of units (pipleines and functions) to load results, by default None
        variables : list, optional
            variables to load, by default None
        iterations : int or list, optional
            iterations to load, by default None
        repetition : int, optional
            repetition to load, by default None
        experiment_id : str, optional
            experiment id to load, by default None
        configs : dict, optional
            specify keys and corresponding values to load results, by default None
        aliases : dict, optional
            the same as `configs` but specify aliases of parameters, by default None
        use_alias : bool, optional
            use aliases for parameter values, by default True
        concat_config : bool, optional
            concatenate all config options into one string and store it in 'config' column, by default False
        drop_columns : bool, optional
            remove parameters columns if `concat_config=True`, by default True
        kwargs : dict
            kwargs will be interpreted as config paramter

        Returns
        -------
        pandas.DataFrame or dict
            dataframe columns: iteration, name (of pipeline/function), repetition, update
            (is used if domain was dynamically updated), and column(-s) for config.
            Also it will have column for each variable of pipeline
            and output of the function that was saved as a result of the research.
        """
        return self._load_df(*args, **kwargs)

    def load_artifactes(self, *args, **kwargs):
        """ Load paths to artifactes for experiments.

        Parameters
        ----------
        name : str, optional
            name of artifact to load, by default None. Can be used with iteration as prefix/postfix
            (see `format` parameter)
        iterations : int or list, optional
            iterations to load, by default None
        repetition : int, optional
            repetition to load, by default None
        experiment_id : str or list, optional
            experiment id to load, by default None
        configs : dict, optional
            specify keys and corresponding values to load paths, by default None
        aliases : dict, optional
            the same as `configs` but specify aliases of parameters, by default None
        use_alias : bool, optional
            use aliases for parameter values, by default True
        concat_config : bool, optional
            concatenate all config options into one string and store it in 'config' column, by default False
        drop_columns : bool, optional
            remove parameters columns if `concat_config=True`, by default True
        format : str, optional
            one of '_{}', '/{}' (iteration as postfix for name), '{}_', '{}/' (iteration as prefix for name),
            by default None
        kwargs : dict
            kwargs will be interpreted as config paramter

        Returns
        -------
        pandas.DataFrame or dict
            dataframe columns: iteration, name (of pipeline/function), repetition, update
            (is used if domain was dynamically updated), column for config, artifact_path (full path to artefact
            including file name), filename.
        """
        return self._load_artifactes(*args, **kwargs)

    def filter_configs(self, repetition=None, experiment_id=None, configs=None, aliases=None, **kwargs):
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
        if len(kwargs) > 0:
            aliases = kwargs if aliases is None else {**aliases, **kwargs}

        if experiment_id is not None:
            _configs = {id: config for id, config in self.configs.items() if id in to_list(experiment_id)}
        else:
            _configs = self.configs

        if configs is None and aliases is None and repetition is None:
            return _configs

        if repetition is not None:
            repetition = {'repetition': repetition}
        else:
            repetition = dict()

        if configs is None and aliases is None:
            configs = dict()

        result = {}
        for experiment_id, supconfig in _configs.items():
            if configs is not None:
                configs.update(repetition)
                _config = supconfig.config()
                if all(item in _config.items() for item in configs.items()):
                    result[experiment_id] = supconfig
            else:
                _config = supconfig.alias()
                aliases.update(repetition)
                if all(item in _config.items() for item in aliases.items()):
                    result[experiment_id] = supconfig
        return result

    def _load_df(self, names=None, variables=None, iterations=None, repetition=None, experiment_id=None,
                 configs=None, aliases=None, use_alias=True, concat_config=False, drop_columns=True, **kwargs):
        _configs = self.filter_configs(repetition, experiment_id, configs, aliases, **kwargs)

        names = to_list(names or self.names)
        variables = to_list(variables or self.variables)
        iterations = to_list(iterations)

        all_results = []
        for _experiment_id, config_alias in _configs.items():
            _repetition = config_alias.pop_config('repetition').config()['repetition']
            _update = config_alias.pop_config('update').config()['update']
            path = os.path.join(self.path, 'results', _experiment_id)

            for unit in names:
                files = glob.glob(glob.escape(os.path.join(path, unit)) + '_[0-9]*')
                files = self._sort_files(files, iterations)
                if len(files) != 0:
                    res = []
                    for filename, iterations_to_load in files.items():
                        with open(filename, 'rb') as file:
                            res.append(self._slice_file(dill.load(file), iterations_to_load, variables))
                    res = self._concat(res, variables)

                    config_alias.pop_config('_dummy')
                    res = self._append_config(res, config_alias, concat_config, use_alias, drop_columns,
                                              repetition=_repetition, update=_update, name=unit)
                    all_results.append(pd.DataFrame(res))
        return pd.concat(all_results, sort=False).reset_index(drop=True) if len(all_results) > 0 else pd.DataFrame(None)

    def _load_artifactes(self, names=None, iterations=None, repetition=None, experiment_id=None, configs=None,
                         aliases=None, use_alias=True, concat_config=False, drop_columns=True,
                         format=None, **kwargs):
        _configs = self.filter_configs(repetition, experiment_id, configs, aliases, **kwargs)
        iterations = to_list(iterations or '*')
        names = to_list(names or '*')

        all_results = []
        for _experiment_id, config_alias in _configs.items():
            _repetition = config_alias.pop_config('repetition').config()['repetition']
            _update = config_alias.pop_config('update').config()['update']
            path = os.path.join(self.path, 'results', _experiment_id)
            for name in names:
                res = {}
                if format in ['_{}', '/{}']:
                    name = name + format
                elif format in ['{}_', '{}/']:
                    name = format + name
                for i in iterations:
                    filenames = []
                    for filename in glob.glob(os.path.join(path, name.format(i))):
                        if not any([os.path.basename(filename).startswith(unit_name) for unit_name in self.names]):
                            filenames.append(filename)
                res.update({'artifact_path': filenames})
                res.update({'filename': [os.path.basename(item) for item in filenames]})
                res.update({'experiment_id': [_experiment_id] * len(filenames)})
                res = self._append_config(res, config_alias, concat_config, use_alias, drop_columns,
                                          repetition=_repetition, update=_update)
                all_results.append(pd.DataFrame(res))
        return pd.concat(all_results, sort=False).reset_index(drop=True) if len(all_results) > 0 else pd.DataFrame(None)

    def _append_config(self, res, config_alias, concat_config, use_alias, drop_columns, **kwargs):
        length = self._fix_length(res)
        if concat_config:
            res['config'] = config_alias.alias(as_string=True)
        if use_alias:
            if not concat_config or not drop_columns:
                res.update(config_alias.alias(as_string=False))
        else:
            config = config_alias.config()
            config = {k: [v] * length for k, v in config.items()}
            res.update(config)
        res.update(kwargs)
        return res

    def _load_configs(self, experiment_id=None):
        configs = {}
        for filename in glob.glob(os.path.join(self.path, 'configs', experiment_id or '*')):
            _experiment_id = os.path.split(filename)[-1]
            with open(filename, 'rb') as f:
                configs[_experiment_id] = dill.load(f)
        return configs

    def _sort_files(self, files, iterations):
        files = {file: int(file.split('_')[-1]) for file in files}
        files = OrderedDict(sorted(files.items(), key=lambda x: x[1]))
        result = []
        start = 0
        iterations = [item for item in iterations if item is not None]
        for name, end in files.items():
            if len(iterations) == 0:
                intersection = np.arange(start, end)
            else:
                intersection = np.intersect1d(iterations, np.arange(start, end))
            if len(intersection) > 0:
                result.append((name, intersection))
            start = end
        return OrderedDict(result)

    def _slice_file(self, dumped_file, iterations_to_load, variables):
        iterations = dumped_file['iteration']
        if len(iterations) > 0:
            elements_to_load = np.array([np.isin(it, iterations_to_load) for it in iterations])
            res = OrderedDict()
            for variable in ['iteration', 'experiment_id', *variables]:
                if variable in dumped_file:
                    res[variable] = np.array(dumped_file[variable])[elements_to_load]
        else:
            res = None
        return res

    def _concat(self, results, variables):
        res = {key: [] for key in [*variables, 'iteration', 'experiment_id']}
        for chunk in results:
            if chunk is not None:
                for key, values in res.items():
                    if key in chunk:
                        values.extend(chunk[key])
        return res

    def _fix_length(self, chunk):
        """ Pad arrays in the dict with nans to the same length. """
        max_len = max([len(value) if isinstance(value, (list, tuple, pd.Series)) else 1 for value in chunk.values()])
        for value in chunk.values():
            if len(value) < max_len:
                value.extend([np.nan] * (max_len - len(value)))
        return max_len

    def _get_description(self):
        with open(os.path.join(self.path, 'description', 'research.json'), 'r') as file:
            return json.load(file)

    def get_config(self, experiment_id=None):
        """ Get configs by experiment_id. """
        res = {}
        for path in glob.glob(os.path.join(self.path, 'configs', experiment_id or '*')):
            with open(path, 'rb') as f:
                res[os.path.basename(path)] = dill.load(f)
        if experiment_id:
            res = res[experiment_id]
        return res
