""" Classes Research and auxiliary classes for multiple experiments. """

import os
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

from .results import Results
from .distributor import Distributor
from .workers import PipelineWorker
from .domain import Domain, Option, ConfigAlias
from .job import Job
from .logger import BaseLogger, FileLogger, PrintLogger, TelegramLogger
from .utils import get_metrics
from .executable import Executable
from .named_expr import RP

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
        self.timeout = None
        self.n_configs = None
        self.n_reps = None
        self.n_configs = None
        self.repeat_each = None
        self.logger = FileLogger()

        # update parameters for config. None or dict with keys (function, params, cache)
        self._update_config = None
        # update parameters for domain. None or dict with keys (function, each)
        self._update_domain = None
        self.n_updates = 0

    def add_pipeline(self, root, branch=None, dataset=None, variables=None,
                     name=None, execute=1, dump='last', run=False, logging=False, **kwargs):
        """ Add new pipeline to research. Pipeline can be divided into root and branch. In that case root pipeline
        will prepare batch that can be used by different branches with different configs.

        Parameters
        ----------
        root : Pipeline
            a pipeline to execute when the research is run. It must contain `run` action with `lazy=True` or
            `run_later`. Only if `branch` is None, `root` may contain parameters that can be defined by domain.
        branch : Pipeline or None
            a parallelized pipeline to execute within the research.
            Several copies of branch pipeline will be executed in parallel per each batch
            received from the root pipeline. May contain parameters that can be defined by domain,
            all pipelines will have different configs from `Domain`.
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
        kwargs : dict
            parameters that will be added to pipeline config.
            Can be `:class:~.RP`.

            For example,
            if test pipeline imports model from the other pipeline with name `'train'` in Research,
            corresponding parameter in `import_model` must be `C('import_from')` and `add_pipeline`
            must be called with parameter `import_from=RP('train')`.


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

    def add_callable(self, function, *args, returns=None, name=None, execute=1, dump='last',
                     on_root=False, logging=False, **kwargs):
        """ Add function to research.

        Parameters
        ----------
        function : callable
            callable object with parameters: `*args, **kwargs`
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
            iteration when results will be dumped and cleared. Similar to execute.
        on_root : bool
            If True, each `ResearchExecutableUnit` in args and kwargs will be evaluated as a list of values for
            each experiment. If False, will be evaluated as a single value for the current experiment.
        logging : bool
            include execution information to log file or not
        args : list
            args for the function. Can be :class:`~.ResearchNamedExpression`.
        kwargs : dict
            kwargs for the function. Can be :class:`~.ResearchNamedExpression`.

        **How to use experiment**

        Each pipeline and function added to Research is saved as an :class:`~.Executable`.

        Experiment is an `OrderedDict` of all pipelines and functions that were added to Research
        and are running in current Job. Key is a name of `Executable`, value is `Executable`.
        """

        name = name or function.__name__

        if name in self.executables:
            raise ValueError('Executable unit with name {} already exists'.format(name))

        if on_root and returns is not None:
            raise ValueError("If function on root, then it mustn't have returns")

        unit = Executable()
        unit.add_callable(function, *args, name=name, execute=execute, dump=dump, returns=returns,
                          on_root=on_root, logging=logging, **kwargs)
        self.executables[name] = unit

        return self

    def get_metrics(self, pipeline, metrics_var, metrics_name, *args,
                    returns=None, execute=1, dump='last', logging=False, agg='mean', **kwargs):
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
            if None, `returns` will be equal to `metrics_name`
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
        name = pipeline + '_' + metrics_var
        returns = returns or metrics_name
        self.add_callable(get_metrics, *args, name=name, execute=execute, dump=dump, returns=returns,
                          on_root=False, logging=logging, pipeline=RP(pipeline),
                          metrics_var=metrics_var, metrics_name=metrics_name, agg=agg, **kwargs)
        return self

    def init_domain(self, domain=None, n_configs=None, n_reps=1, repeat_each=100):
        """ Add domain of pipeline parameters. Configs from that domain will be generated
        and then substitute into pipelines.

        Parameters
        ----------
        domain : Domain or Option
        """
        if isinstance(domain, Option):
            self.domain = Domain(domain)
        elif domain is None:
            self.domain = Domain(Option('_dummy', [None]))
        else:
            self.domain = domain
        self.n_configs = n_configs
        self.n_reps = n_reps
        self.n_configs = n_configs
        self.repeat_each = repeat_each
        return self

    def update_domain(self, function=None, each='last', n_updates=1, **kwargs):
        """ Add domain update functions or update parameters.

        Parameters
        ----------
        function : callable or None

        each : int or 'last'
            when update method will be called. If 'last', domain will be updated
            when iterator will be finished. If int, domain will be updated with
            that period.
        n_updates : int
            the total number of updates.
        *args, **kwargs :
            update function parameters.
        """
        self._update_domain = {
            'function': function,
            'each': each,
            'kwargs': kwargs
        }
        self.n_updates = n_updates
        return self

    def update_config(self, function, parameters=None, cache=0):
        """ Add function to update config from domain.

        Parameters
        ----------
        function : callable
            function to update config.
        parameters : list of str or None
            if None, the whole config will be passed into function.
        cache : int
            last `cache` calls of function will be cahced.
        """
        self._update_config = {
            'function': function,
            'params': parameters,
            'cache': cache
        }
        return self

    def add_logger(self, logger, **kwargs):
        """ Add custom Logger into Research.

        Parameters
        ----------
        logger : str, BaseLogger class, tuple or list
            if str, it can be 'file', 'print' or 'tg'
            if tuple, pair of str or BaseLogger class and kwargs for them
            if list then of str, BaseLogger class and tuples of them and kwargs
        kwargs :
            initialization parameters for BaseLogger (if `logger` is str or BaseLogger class)
        """
        loggers = [logger] if not isinstance(logger, list) else logger

        self.logger = BaseLogger()

        for item in loggers:
            if not isinstance(item, tuple):
                item = (item, kwargs)
            logger, params = item

            if isinstance(logger, str):
                if logger == 'file':
                    self.logger += FileLogger()
                elif logger == 'print':
                    self.logger += PrintLogger()
                elif logger == 'tg':
                    self.logger += TelegramLogger(**params)
                else:
                    raise ValueError('Unknown logger: {}'.format(logger))
            elif issubclass(logger, BaseLogger):
                self.logger += logger(**params)
            else:
                raise ValueError('Unknown logger: {}'.format(logger))

        return self

    def load_results(self, *args, **kwargs):
        """ Load results of research as pandas.DataFrame or dict (see :meth:`~.Results.load`). """
        return Results(self.name, *args, **kwargs)

    def run(self, n_iters=None, workers=1, branches=1, name=None,
            bar=False, devices=None, worker_class=None, timeout=None, trials=2):
        """ Run research.

        Parameters
        ----------
        n_iters: int or None
            number of iterations for each configuration. If None, wait StopIteration exception for at least
            one executable unit.
        workers : int or list of dicts (Configs)
            If int - number of workers to run pipelines or workers that will run them, `worker_class` will be used.

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
            all devices will be distributed between workwers and branches.
            If you want to use different devices in branches, use expression `C('device')`.

            For example, for :class:`~.TFModel` add `device=C('device')` to model config.
            If None, default gpu configuration will be used
        worker_class : type
            worker class. `PipelineWorker` by default.
        timeout : int
            each job will be killed if it doesn't answer more then that time in minutes
        trials : int
            trials to execute job

        **How does it work**

        At each iteration all pipelines and functions will be executed in the order in which were added.
        If `update_config` callable is defined, each config will be updated by that function and then
        will be passed into each `ExecutableUnit`.
        If `update_domain` callable is defined, domain will be updated with the corresponding function
        accordingly to `each` parameter of `update_domain`.
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
            self.init_domain()

        self.domain.reset_iter()
        self._folder_exists(self.name)
        self.__save()

        self.domain = self.domain * Option('update', [0])
        self.domain.set_iter(n_iters=self.n_configs, n_reps=self.n_reps, repeat_each=self.repeat_each)

        if self.domain.size is None and (self._update_domain is None or self._update_domain['each'] == 'last'):
            warnings.warn("Research will be infinite because has infinite domain and hasn't domain updating",
                          stacklevel=2)

        print("Research {} is starting...".format(self.name))

        jobs_queue = DynamicQueue(self.branches, self.domain, self.n_iters, self.executables,
                                  self.name, self._update_config, self._update_domain, self.n_updates)
        self.logger.eval_kwargs(path=self.name)
        distr = Distributor(self.n_iters, self.workers, self.devices, self.worker_class, self.timeout,
                            self.trials, self.logger)
        distr.run(jobs_queue, bar=self.bar)

        return self

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__.update(d)

    def _get_devices(self, devices):
        n_branches = self.branches if isinstance(self.branches, int) else len(self.branches)
        n_workers = self.workers if isinstance(self.workers, int) else len(self.workers)
        total_n_branches = n_workers * n_branches
        if devices is None:
            devices = [[[None]] * n_branches] * n_workers
        if isinstance(devices, (int, str)):
            devices = [devices]
        if isinstance(devices[0], (int, str)):
            if total_n_branches > len(devices):
                _devices = list(itertools.chain.from_iterable(
                    zip(*itertools.repeat(devices, total_n_branches // len(devices)))
                ))
                devices = _devices + devices[:total_n_branches % len(devices)]
            else:
                devices = devices + devices[:-len(devices) % (total_n_branches)]
            if total_n_branches % len(devices) == 0:
                branches_per_device = total_n_branches // len(devices)
                devices = list(itertools.chain.from_iterable(itertools.repeat(x, branches_per_device) for x in devices))
            if len(devices) % total_n_branches == 0:
                devices_per_branch = len(devices) // total_n_branches
                devices = [
                    [
                        [
                            devices[n_branches * devices_per_branch * i + devices_per_branch * j + k]
                            for k in range(devices_per_branch)
                        ] for j in range(n_branches)
                    ] for i in range(n_workers)
                ]
        if isinstance(devices[0], list):
            def _transform_item(x):
                values = [str(item) if isinstance(item, int) else item for item in x]
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
        """ Save description of the research to folder 'name/description'. """
        dirname = os.path.join(self.name, 'description')
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        with open(os.path.join(self.name, 'description', 'research.dill'), 'wb') as file:
            dill.dump(self, file) # contains `Research` object
        with open(os.path.join(self.name, 'description', 'research.json'), 'w') as file:
            # contains `Research` parameters as json
            file.write(json.dumps(self._json(), default=self._set_default_json))
        with open(os.path.join(self.name, 'description', 'alias.json'), 'w') as file:
            # contains representation of the initial domain
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
        description['executables'] = {key: value.__dict__ for key, value in self.executables.items()}
        return description

    def describe(self):
        pprint.pprint(self.__dict__)

    @classmethod
    def load(cls, name):
        """ Load description of the research from 'name/description'. """
        with open(os.path.join(name, 'description', 'research.dill'), 'rb') as file:
            research = dill.load(file)
            research.loaded = True
            return research

class DynamicQueue:
    """ Queue of tasks that can be changed depending on previous results. """
    def __init__(self, branches, domain, n_iters, executables, research_path, update_config, update_domain, n_updates):
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
        self.n_updates = n_updates
        self.update_idx = 0

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
        """ Total estimated size of queue. """
        if self._domain_size is not None:
            rolling_mean = (pd.Series(self.each_config_produce)
                            .rolling(window=10, min_periods=1)
                            .mean().round().values[-1])
            num = np.sum(self.each_config_produce)
            estimated_num = (self._domain_size - len(self.each_config_produce)) * rolling_mean
            return np.ceil((num +  estimated_num) / self.n_branches)
        return None

    def update(self):
        """ Update domain. """
        if self.n_updates is None or self.update_idx < self.n_updates:
            new_domain = self.domain.update_domain(self.research_path)
            self.update_idx += 1
            if new_domain is not None:
                self.domain = new_domain * Option('update', [self.update_idx])
                self._domain_size = self.domain.size
                if self.update_domain is not None:
                    self.domain.set_update(**self.update_domain)
                self.generator = self._generate_config(self.domain)
                return True
        return False

    def next_jobs(self, n_tasks=1):
        """ Get next `n_tasks` elements of queue. """
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
        """ Stop all workers. """
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
