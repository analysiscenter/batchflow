""" Classes Research and auxiliary classes for multiple experiments. """

import os
import glob
from copy import copy
from collections import OrderedDict
import itertools
from functools import lru_cache
import json
import pprint
import warnings
import dill
import numpy as np
import pandas as pd
import multiprocess as mp

from .distributor import Distributor
from .workers import PipelineWorker
from .domain import Domain, Option, ConfigAlias
from .job import Job
from .utils import get_metrics
from .executable import Executable
from .named_expr import EU

class Research:
    """ Class Research for multiple parallel experiments with pipelines. """
    def __init__(self):
        self.executables = OrderedDict()
        self.loaded = False # TODO: Think about it. Do we need load?
        self.branches = 1
        self.trials = 2
        self.workers = 1
        self.bar = False
        self.name = 'research'
        self.worker_class = PipelineWorker
        self.devices = None
        self.domain = None
        self.n_iters = None
        self.timeout = 5

        # update parameters for config. None or dict with keys (function, params, cache)
        self._update_config = None
        # update parameters for domain. None or dict with keys (funct, each)
        self._update_domain = None

    def add_pipeline(self, root, branch=None, dataset=None, variables=None,
                     name=None, execute=1, dump='last', run=False, logging=False, **kwargs):
        """ Add new pipeline to research. Pipeline can be divided into root and branch. In that case root pipeline
        will prepare batch that can be used by different branches with different configs.

        Parameters
        ----------
        root : Pipeline
            a pipeline to execute when the research is run. It must contain `run` action with `lazy=True`.
            Only if `branch` is None, `root` may contain parameters that can be defined by domain.
        branch : Pipeline or None
            a parallelized pipeline to execute within the research.
            Several copies of branch pipeline will be executed in parallel per each batch
            received from the root pipeline.
            May contain parameters that can be defined by domain.
        dataset : Dataset or None
            dataset that will be used with pipelines. If None, root or branch must contain dataset.
        variables : str, list of str or None
            names of pipeline variables to save after each iteration into results. All of them must be
            defined in `root` if `branch` is None or be defined in `branch` if `branch` is not None.
            If None, pipeline will be executed without any dumping.
        name : str
            pipeline name. If None, pipeline will have name `pipeline_{index}`
        execute : int, str or list of int and str
            If `'last'`, pipeline will be executed just at last iteration (if `iteration + 1 == n_iters`
            or `StopIteration` was raised)

            If positive int, pipeline will be executed each `step` iterations.

            If str, must be `'#{it}'` or `'last'` where it is int,
            the pipeline will be executed at this iteration (zero-based)

            If list, must be list of int or str described above
        dump : int, str or list of int or str
            iteration when results will be dumped and cleared. Similar to `execute`
        run : bool
            if False then `.next_batch()` will be applied to pipeline, else `.run()` and then `.reset("iter")`.
        logging : bool
            include execution information to log file or not
        kwargs :
            parameters in pipeline config that depends on the names of the other pipeline.

            For example,
            if test pipeline imports model from the other pipeline with name `'train'` in Research,
            corresponding parameter in `import_model` must be `C('import_from')` and `add_pipeline`
            must be called with parameter `import_from='train'`.


        **How to define changing parameters**

        All parameters in `root` or `branch` that are defined in domain should be defined
        as `C('parameter_name')`. Corresponding parameter in domain must have the same `'parameter_name'`.
        """
        name = name or 'pipeline_' + str(len(self.executables) + 1)

        if name in self.executables:
            raise ValueError('Executable unit with name {} already exists'.format(name))

        unit = Executable()
        unit.add_pipeline(root, name, branch, dataset, variables, execute, dump, run, logging, **kwargs)
        self.executables[name] = unit
        return self

    def get_metrics(self, pipeline, metrics_var, metrics_name,
                    returns=None, execute=1, dump='last', logging=False):
        """ Evaluate metrics.

        Parameters
        ----------
        pipeline : str
            pipeline name
        metrics_var : str
            pipeline variable which accumulate metrics
        metrics_name : str or list of str
            metrics to evaluate
        returns : str, list of str or None
            names to save metrics into results
            if None, `function` will be executed without any saving results and dumping
        execute : int, str or list of int and str
            If `'last'`, metrics will be gathered just at last iteration (if `iteration + 1 == n_iters`
            or `StopIteration` was raised)

            If positive int, metrics will be gathered each `step` iterations.

            If str, must be `'#{it}'` or `'last'` where it is int,
            metrics will be gathered at this iteration (zero-based)

            If list, must be list of int or str described above
        dump : int, str or list of int or str
            iteration when results will be dumped and cleared. Similar to execute
        logging : bool
            include execution information to log file or not
        """
        name = pipeline + '_metrics'
        self.add_function(get_metrics, returns, name, execute, dump,
                          False, logging, experiment=EU(), pipeline=pipeline,
                          metrics_var=metrics_var, metrics_name=metrics_name)
        return self

    def add_function(self, function, returns=None, name=None, execute=1, dump='last',
                     on_root=False, logging=False, *args, **kwargs):
        """ Add function to research.

        Parameters
        ----------
        function : callable
            callable object with following parameters:
                experiment : `OrderedDict` of Executable objects
                    all pipelines and functions that were added to Research
                iteration : int
                    iteration when function is called
                **args, **kwargs
        returns : str, list of str or None
            names for function returns to save into results
            if None, `function` will be executed without any saving results and dumping
        name : str (default None)
            function name. If None, a function will have name `func_{index}`
        execute : int, str or list of int and str
            If `'last'`, function will be executed just at last iteration (if `iteration + 1 == n_iters`
            or `StopIteration` was raised)

            If positive int, function will be executed each `step` iterations.

            If str, must be `'#{it}'` or `'last'` where it is int,
            the function will be executed at this iteration (zero-based)

            If list, must be list of int or str described above
        dump : int, str or list of int or str
            iteration when results will be dumped and cleared. Similar to execute
        on_root : bool
            if False, function will be called with parameters `(iteration, experiment, *args, **kwargs)`,
            else with `(iteration, experiments, *args, **kwargs)` where `experiments` is a list of single experiments
        logging : bool
            include execution information to log file or not
        args : list
            args for the function
        kwargs : dict
            kwargs for the function


        **How to use experiment**

        Each pipeline and function added to Research is saved as an :class:`~.Executable`.

        Experiment is an `OrderedDict` of all pipelines and functions that were added to Research
        and are running in current Job. Key is a name of `Executable`, value is `Executable`.
        """

        name = name or 'func_' + str(len(self.executables) + 1)

        if name in self.executables:
            raise ValueError('Executable unit with name {} was alredy existed'.format(name))

        if on_root and returns is not None:
            raise ValueError("If function on root, then it mustn't have returns")

        unit = Executable()
        unit.add_function(function, name, execute, dump,
                          returns, on_root, logging, *args, **kwargs)
        self.executables[name] = unit

        return self

    def init_domain(self, domain):
        """ Add domain of pipeline parameters. Configs from that domain will be generated
        and then substitute into pipelines.

        Parameters
        ----------
        domain : Domain or Option
            if dict it should have items parameter_name: list of values.
        """
        self.domain = domain
        return self

    def update_domain(self, function=None, each='last'):
        self._update_domain = {
            'function': function,
            'each': each
        }
        return self

    def update_config(self, function, parameters=None, cache=0):
        """ Add function to update config from domain.

        Parameters
        ----------
        domain : dict, Domain or Option
            if dict it should have items parameter_name: list of values.
        """
        self._update_config = {
            'function': function,
            'params': parameters,
            'cache': cache
        }
        return self

    def load_results(self, *args, **kwargs):
        """ Load results of research as pandas.DataFrame or dict (see Results.load). """
        return Results(research=self).load(*args, **kwargs)

    def run(self, n_iters=None, workers=1, branches=1, name=None,
            bar=False, devices=None, worker_class=None, timeout=5, trials=2):

        """ Run research.

        Parameters
        ----------
        n_iters: int or None
            number of iterations for each configurations. If None, wait StopIteration exception for at least
            one pipeline.
        workers : int or list of dicts (Configs)
            If int - number of workers to run pipelines or workers that will run them, `PipelineWorker` will be used.

            If list of dicts (Configs) - list of additional configs which will be appended to configs from tasks.

            Each element corresponds to one worker.
        branches: int or list of dicts (Configs)
            Number of different branches with different configs with the same root. Each branch will use the same batch
            from `root`. Pipelines will be executed in different threads.

            If int - number of pipelines with different configs that will use the same prepared batch
            from `root`.

            If list of dicts (Configs) - list of dicts with additional configs to each pipeline.
        name : str or None
            name folder to save research. By default is 'research'.
        bar : bool or callable
            Whether to show a progress bar.
            If callable, it must have the same signature as `tqdm`.
        devices : str, list or None
            all devices available for the research.

            Must be of length 1 or be divisible
            by the number of workers.

            If is divisible by the number of workers then
            `length / n_workers` must be 1 or be divisible by the number of branches. If you want to use different
            devices in branches, use expression `C('device')`.

            For example, for :class:`~.TFModel` add `device=C('device')` to model config.
            if None, default gpu configuration will be used
        timeout : int
            each job will be killed if it doesn't answer more then that time in minutes
        trials : int
            trials to execute job

        **How does it work**

        At each iteration all pipelines and functions will be executed in the order in which were added.
        """
        if self.loaded:
            print("Starting loaded research. All parameters passed to run except name, bar and devices are ignored.\n",
                  "If `devices` is not provided it will be inherited")
            if devices is not None:
                self.devices = self._get_devices(devices)
        else:
            self.n_iters = n_iters
            self.workers = workers
            self.branches = branches
            self.devices = self._get_devices(devices)
            self.worker_class = worker_class or PipelineWorker
            self.timeout = timeout
            self.trials = trials

        self.name = name or self.name
        self.bar = bar

        if self.domain is None:
            self.domain = Domain(Option('_dummy', [None]))
            self.domain.set_iterator(brute_force=True, n_iters=None, n_reps=1)

        if self.domain.size is None and (self._update_domain is None or self._update_domain['each'] == 'last'):
            warnings.warn("Research will be infinite because has infinite domain and hasn't domain updating",
                          stacklevel=2)
        self._folder_exists(self.name)

        print("Research {} is starting...".format(self.name))

        self.__save()

        jobs_queue = DynamicQueue(self.branches, self.domain, self.n_iters, self.executables,
                                  self.name, self._update_config, self._update_domain)

        distr = Distributor(self.n_iters, self.workers, self.devices, self.worker_class, self.timeout, self.trials)
        distr.run(jobs_queue, dirname=self.name, bar=self.bar)

        return self

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__.update(d)

    def _get_devices(self, devices):
        # TODO: process `device` to simplify nested list construction
        n_branches = self.branches if isinstance(self.branches, int) else len(self.branches)
        n_workers = self.workers if isinstance(self.workers, int) else len(self.workers)
        if devices is None:
            devices = [[[None]] * n_branches] * n_workers
        if isinstance(devices, (int, str)):
            devices = [devices]
        if isinstance(devices[0], (int, str)):
            if n_workers * n_branches % len(devices) == 0:
                branches_per_device = n_workers * n_branches // len(devices)
                devices = list(itertools.chain.from_iterable(itertools.repeat(x, branches_per_device) for x in devices))
            if len(devices) % (n_workers * n_branches) == 0:
                devices_per_branch = len(devices) // (n_workers * n_branches)
                devices = [
                    [
                        [
                            devices[n_branches * devices_per_branch * i + devices_per_branch * j + k]
                            for k in range(devices_per_branch)
                        ] for j in range(n_branches)
                    ] for i in range(n_workers)
                ]
            else:
                raise ValueError('n_workers * n_branches must be divisible by n_devices or vice versa')
        if isinstance(devices[0], list):
            def _transform_item(x):
                if len(x) > 1:
                    values = [str(item) for item in x]
                else:
                    values = str(x[0])
                return dict(device=values) if x is not None else dict()

            devices = [[_transform_item(branch_config) for branch_config in worker_config] for worker_config in devices]
        return devices

    @staticmethod
    def _folder_exists(name):
        if not os.path.exists(name):
            os.makedirs(name)
            config_path = os.path.join(name, 'configs')
            if not os.path.exists(config_path):
                os.makedirs(config_path)
        else:
            raise ValueError(
                "Research with name '{}' already exists".format(name)
            )

    def __save(self):
        """ Save description of the research to folder name/description. """
        dirname = os.path.join(self.name, 'description')
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        with open(os.path.join(self.name, 'description', 'research.dill'), 'wb') as file:
            dill.dump(self, file)
        with open(os.path.join(self.name, 'description', 'research.json'), 'w') as file:
            file.write(json.dumps(self._json(), default=self._set_default_json))
        with open(os.path.join(self.name, 'description', 'alias.json'), 'w') as file:
            file.write(json.dumps(str(self.domain), default=self._set_default_json))

    def _set_default_json(self, obj):
        try:
            x = json.dumps(obj)
        except TypeError:
            x = str(obj)
        return x

    def _json(self):
        description = copy(self.__dict__)
        description['domain'] = str(self.domain)
        return description

    def describe(self):
        pprint.pprint(self.__dict__)

    @classmethod
    def load(cls, name):
        """ Load description of the research from name/description. """
        with open(os.path.join(name, 'description', 'research.dill'), 'rb') as file:
            research = dill.load(file)
            research.loaded = True
            return research

class DynamicQueue:
    def __init__(self, branches, domain, n_iters, executables, research_path, update_config, update_domain):
        self.branches = branches
        self.domain = domain
        self.n_iters = n_iters
        self.executables = executables
        self.research_path = research_path

        if update_config is not None and update_config['cache'] > 0:
            update_config['function'] = lru_cache(maxsize=update_config['cache'])(update_config['function'])

        self.update_config = update_config

        self.n_branches = branches if isinstance(branches, int) else len(branches)

        self.domain = domain
        self.update_domain = update_domain
        self._domain_size = self.domain.size

        if self.update_domain is not None:
            self.domain.set_update(**self.update_domain)

        self.generator = self._generate_config(self.domain)

        self._queue = mp.JoinableQueue()

        self.generated_jobs = 0

    def _generate_config(self, domain):
        self.each_config_produce = []
        while True:
            try:
                config_from_domain = next(domain)
                config_from_func = dict()
                if self.update_config is not None:
                    if self.update_config['params'] is None:
                        config_from_func = self.update_config['function'](config_from_domain.config())
                    else:
                        _config_slice = {key: config_from_domain.config().get(key)
                                         for key in self.update_config['params']}
                        config_from_func = self.update_config['function'](**_config_slice)
                config_from_func = config_from_func if isinstance(config_from_func, list) else [config_from_func]
                self.each_config_produce.append(len(config_from_func))
                for config in config_from_func:
                    yield (config_from_domain, ConfigAlias(config.items()))
            except StopIteration:
                break

    @property
    def total(self):
        if self._domain_size is not None:
            rolling_mean = pd.Series(self.each_config_produce).rolling(window=10, min_periods=1).mean().round().values[-1]
            num = np.sum(self.each_config_produce)
            estimated_num = (self._domain_size - len(self.each_config_produce)) * rolling_mean
            return np.ceil((num +  estimated_num) / self.n_branches)
        else:
            return None

    def update(self):
        new_domain = self.domain.update_domain(self.research_path)
        if new_domain is not None:
            self.domain = new_domain
            self._domain_size = self.domain.size
            if self.update_domain is not None:
                self.domain.set_update(**self.update_domain)
            self.generator = self._generate_config(self.domain)
            return True
        else:
            return False

    def next_jobs(self, n_tasks=1):
        configs = []
        for i in range(n_tasks):
            branch_tasks = []
            try:
                for _ in range(self.n_branches):
                    branch_tasks.append(next(self.generator))
                configs.append(branch_tasks)
            except StopIteration:
                if len(branch_tasks) > 0:
                    configs.append(branch_tasks)
                break
        for i, config in enumerate(configs):
            self.put((self.generated_jobs + i,
                      Job(self.executables, self.n_iters, config, self.branches, self.research_path)))

        n_tasks = len(configs)
        self.generated_jobs += n_tasks

        return n_tasks

    def stop_workers(self, n_workers):
        for _ in range(n_workers):
            self.put(None)

    def join(self):
        self._queue.join()

    def get(self):
        return self._queue.get()

    def put(self, value):
        self._queue.put(value)

    def task_done(self):
        self._queue.task_done()

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
            else:
                _config = supconfig.alias()
            if all(item in _config.items() for item in alias.items()):
                result.append(_config)
        self.configs = result


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
            .add_function(accuracy, returns='accuracy', name='test_accuracy',
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
            names = list(self.research.executables.keys())

        if variables is None:
            variables = [variable for unit in self.research.executables.values() for variable in unit.variables]

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
                        all_results.append(
                            pd.DataFrame({
                                'name': unit,
                                **res
                            })
                            )
        return pd.concat(all_results, sort=False).reset_index(drop=True) if len(all_results) > 0 else pd.DataFrame(None)
