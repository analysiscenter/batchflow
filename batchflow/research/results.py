""" Research results class """

import os
from collections import OrderedDict
import glob
import json
import dill
import pandas as pd

class Results:
    """ Class for dealing with results of research

    Parameters
    ----------
    path : str
        path to root folder of research
    names : str, list or None
        names of units (pipleines and functions) to load
    variables : str, list or None
        names of variables to load
    iterations : int, list or None
        iterations to load
    repetition : int
        index of repetition to load
    configs, aliases : dict, Config, Option, Domain or None
        configs to load
    use_alias : bool
        if True, use alias for model name, else use its full name.
        Defaults to True
    concat_config : bool
        if True, concatenate all config options into one string and store
        it in 'config' column, else use separate column for each option.
        Defaults to False
    drop_columns : bool
        used only if `concat_config=True`. Drop or not columns with options and
        leave only concatenated config.
    kwargs : dict
        kwargs will be interpreted as config paramter

    Returns
    -------
    pandas.DataFrame or dict
        will have columns: iteration, name (of pipeline/function)
        and column for config. Also it will have column for each variable of pipeline
        and output of the function that was saved as a result of the research.

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
    def __init__(self, path, *args, **kwargs):
        self.path = path
        self.description = self._get_description()
        self.configs = None
        self.df = self._load(*args, **kwargs)

    def _get_list(self, value):
        if not isinstance(value, list):
            value = [value]
        return value

    def _sort_files(self, files, iterations):
        files = {file: int(file.split('_')[-1]) for file in files}
        files = OrderedDict(sorted(files.items(), key=lambda x: x[1]))
        result = []
        start = 0
        iterations = [item for item in iterations if item is not None]
        for name, end in files.items():
            if len(iterations) == 0:
                intersection = pd.np.arange(start, end)
            else:
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
            for variable in ['iteration', 'sample_index', *variables]:
                if variable in dumped_file:
                    res[variable] = pd.np.array(dumped_file[variable])[elements_to_load]
        else:
            res = None
        return res

    def _concat(self, results, variables):
        res = {key: [] for key in [*variables, 'iteration', 'sample_index']}
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

    def _filter_configs(self, config=None, alias=None, repetition=None):
        result = None
        if config is None and alias is None and repetition is None:
            raise ValueError('At least one of parameters config, alias and repetition must be not None')
        result = []
        if repetition is not None:
            repetition = {'repetition': repetition}
        else:
            repetition = dict()

        if config is None and alias is None:
            config = dict()

        for supconfig in self.configs:
            if config is not None:
                config.update(repetition)
                _config = supconfig.config()
                if all(item in _config.items() for item in config.items()):
                    result.append(supconfig)
            else:
                _config = supconfig.alias()
                alias.update(repetition)
                if all(item in _config.items() for item in alias.items()):
                    result.append(supconfig)
        self.configs = result

    def _get_description(self):
        with open(os.path.join(self.path, 'description', 'research.json'), 'r') as file:
            return json.load(file)

    def _load(self, names=None, variables=None, iterations=None, repetition=None, sample_index=None,
              configs=None, aliases=None, use_alias=True, concat_config=False, drop_columns=True, **kwargs):
        self.configs = []
        for filename in glob.glob(os.path.join(self.path, 'configs', '*')):
            with open(filename, 'rb') as f:
                self.configs.append(dill.load(f))

        if len(kwargs) > 0:
            if configs is None:
                configs = kwargs
            else:
                configs.update(kwargs)

        if configs is not None:
            self._filter_configs(config=configs, repetition=repetition)
        elif aliases is not None:
            self._filter_configs(alias=aliases, repetition=repetition)
        elif repetition is not None:
            self._filter_configs(repetition=repetition)

        if names is None:
            names = list(self.description['executables'].keys())

        if variables is None:
            variables = [variable
                         for unit in self.description['executables'].values()
                         for variable in unit['variables']
                        ]

        names = self._get_list(names)
        variables = self._get_list(variables)
        iterations = self._get_list(iterations)

        all_results = []
        for config_alias in self.configs:
            alias_str = config_alias.alias(as_string=True)
            _repetition = config_alias.pop_config('repetition')
            _update = config_alias.pop_config('update')
            path = os.path.join(self.path, 'results', alias_str)

            for unit in names:
                sample_folders = glob.glob(os.path.join(glob.escape(path), sample_index or '*'))
                for sample_folder in sample_folders:
                    files = glob.glob(glob.escape(os.path.join(sample_folder, unit)) + '_[0-9]*')
                    files = self._sort_files(files, iterations)
                    if len(files) != 0:
                        res = []
                        for filename, iterations_to_load in files.items():
                            with open(filename, 'rb') as file:
                                res.append(self._slice_file(dill.load(file), iterations_to_load, variables))
                        res = self._concat(res, variables)
                        self._fix_length(res)

                        config_alias.pop_config('_dummy')
                        if concat_config:
                            res['config'] = config_alias.alias(as_string=True)
                        if use_alias:
                            if not concat_config or not drop_columns:
                                res.update(config_alias.alias(as_string=False))
                        else:
                            res.update(config_alias.config())
                        res.update({'repetition': _repetition.config()['repetition']})
                        res.update({'update': _update.config()['update']})
                        all_results.append(
                            pd.DataFrame({
                                'name': unit,
                                **res
                            })
                            )
        return pd.concat(all_results, sort=False).reset_index(drop=True) if len(all_results) > 0 else pd.DataFrame(None)
