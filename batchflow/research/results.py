import os
from collections import OrderedDict
import dill
import glob
import json
import pandas as pd

class Results:
    """ Class for dealing with results of research

    Parameters
    ----------
    path : str
        path to root folder of research
    """
    def __init__(self, path):
        self.path = path
        self.description = self._get_description()
        self.configs = None

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

    def _filter_configs(self, config=None, alias=None):
        result = None
        if config is None and alias is None:
            raise ValueError('At least one of parameters config and alias must be not None')
        result = []
        for supconfig in self.configs:
            if config is not None:
                _config = supconfig.config()
                if all(item in _config.items() for item in config.items()):
                    result.append(supconfig)
            else:
                _config = supconfig.alias()
                if all(item in _config.items() for item in alias.items()):
                    result.append(supconfig)
        self.configs = result
    
    def _get_description(self):
        with open(os.path.join(self.path, 'description', 'research.json'), 'r') as file:
            return json.load(file)

    def load(self, names=None, variables=None, iterations=None,
             configs=None, aliases=None, use_alias=True, concat_config=False):
        """ Load results as pandas.DataFrame.

        Parameters
        ----------
        names : str, list or None
            names of units (pipleines and functions) to load
        variables : str, list or None
            names of variables to load
        iterations : int, list or None
            iterations to load
        configs, aliases : dict, Config, Option, Domain or None
            configs to load
        use_alias : bool
            if True, the resulting DataFrame will have one column with alias, else it will
            have column for each option in domain

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
        self.configs = []
        for filename in glob.glob(os.path.join(self.path, 'configs', '*')):
            with open(filename, 'rb') as f:
                self.configs.append(dill.load(f))
        if configs is not None:
            self._filter_configs(config=configs)
        elif aliases is not None:
            self._filter_configs(alias=aliases)

        if names is None:
            names = list(self.description['executables'].keys())

        if variables is None:
            variables = [variable for unit in self.description['executables'].values() for variable in unit['variables']]

        self.names = self._get_list(names)
        self.variables = self._get_list(variables)
        self.iterations = self._get_list(iterations)

        all_results = []

        for config_alias in self.configs:
            alias = config_alias.alias(as_string=False)
            config = config_alias.config()
            alias_str = config_alias.alias(as_string=True)
            path = os.path.join(self.path, 'results', alias_str)

            for unit in self.names:
                sample_folders = glob.glob(os.path.join(glob.escape(path), '*'))
                for sample_folder in sample_folders:
                    files = glob.glob(os.path.join(sample_folder, unit + '_[0-9]*'))
                    files = self._sort_files(files, self.iterations)
                    if len(files) != 0:
                        res = []
                        for filename, iterations_to_load in files.items():
                            with open(filename, 'rb') as file:
                                res.append(self._slice_file(dill.load(file), iterations_to_load, self.variables))
                        res = self._concat(res, self.variables)
                        self._fix_length(res)
                        if '_dummy' not in alias:
                            if use_alias:
                                if concat_config:
                                    res['config'] = alias
                                else:
                                    res.update(alias)
                            else:
                                res.update(config)
                        if 'repetition' in config:
                            res.update({'repetition': config['repetition']})
                        all_results.append(
                            pd.DataFrame({
                                'name': unit,
                                **res
                            })
                            )
        return pd.concat(all_results, sort=False).reset_index(drop=True) if len(all_results) > 0 else pd.DataFrame(None)
