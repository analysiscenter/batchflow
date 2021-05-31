import os
import datetime
import csv
import itertools
import subprocess
import re
import functools
import multiprocess as mp
import dill
import glob
import tqdm
from collections import OrderedDict

import numpy as np
import pandas as pd
import warnings

from .domain import Domain
from .distributor import Distributor
from .experiment import Experiment, Executor
from .utils import create_logger, to_list

class Research:
    def __init__(self, name='research', domain=None, experiment=None, n_configs=None, n_reps=1, repeat_each=None):
        """ Research is an instrument to run multiple parallel experiments with different combinations of
        parameters called experiment configs.

        Parameters
        ----------
        name : str, optional
            name of the research and corresponding folder to store results, by default 'research'
        domain : Domain or Option, optional
            grid of parameters to produce experiment configs, by default None
        experiment : Experiment, optional
            description of the experiment (see :class:`experiment.Experiment`), by default None
        n_configs : int, optional
            the number of configs to get from domain (see `n_items` of :meth:`domain.Domain.set_iter`), by default None
        n_reps : int, optional
            the number of repetitions for each config (see :meth:`domain.Domain.set_iter`), by default 1
        repeat_each : int, optional
            see :meth:`domain.Domain.set_iter`, by default 100
        """
        self.name = name
        self.domain = Domain(domain)
        self.experiment = experiment or Experiment()
        self.n_configs = n_configs
        self.n_reps = n_reps
        self.repeat_each = repeat_each
        self._env = dict()

    def add_instance(self, *args, **kwargs):
        self.experiment.add_instance(*args, **kwargs)
        return self

    def add_callable(self, *args, **kwargs):
        self.experiment.add_callable(*args, **kwargs)
        return self

    def add_generator(self, *args, **kwargs):
        self.experiment.add_generator(*args, **kwargs)
        return self

    def add_pipeline(self, *args, **kwargs):
        self.experiment.add_pipeline(*args, **kwargs)
        return self

    def save(self, *args, **kwargs):
        self.experiment.save(*args, **kwargs)
        return self

    def dump(self, *args, **kwargs):
        self.experiment.dump(*args, **kwargs)
        return self

    def update_domain(self, function, when, **kwargs):
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
        kwargs :
            update function parameters.
        """
        self.domain.set_update(function, when, **kwargs)
        return self

    def load_results(self, *args, **kwargs):
        """ Load results of research as pandas.DataFrame or dict (see :meth:`~.Results.load`). """
        return Results(self.name, *args, **kwargs)

    def attach_env_meta(self, **kwargs):
        """ Save the information about the current state of project repository: commit, diff, status and others.

        Parameters
        ----------
        kwargs : dict
            dict where values are bash commands and keys are names of files to save output of the command.
            Results will be stored in `env` subfolder of the research.
        """
        commands = {
            'commit': "git log --name-status HEAD^..HEAD",
            'diff': 'git diff',
            'status': 'git status',
            **kwargs
        }

        for filename, command in commands.items():
            process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
            output, _ = process.communicate()
            result = re.sub('"image/png": ".*?"', '"image/png": "..."', output.decode('utf'))
            if self.dump_results:
                if not os.path.exists(os.path.join(self.name, 'env')):
                    os.makedirs(os.path.join(self.name, 'env'))
                with open(os.path.join(self.name, 'env', filename + '.txt'), 'w') as file:
                    print(result, file=file)
            else:
                self._env[filename] = result

    @property
    def env(self):
        env = dict()
        if self.dump_results:
            filenames = glob.glob(os.path.join(self.name, 'env', '*'))
            for filename in filenames:
                name = os.path.splitext(os.path.basename(filename))[0]
                with open(filename, 'r') as file:
                    env[name] = file.read().strip()
            return env
        else:
            return self._env

    def get_devices(self, devices):
        """ Return list if lists. Each sublist consists of devices for each branch. """ #TODO extend
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
                return values if x is not None else []

            devices = [[_transform_item(branch_config) for branch_config in worker_config] for worker_config in devices]
        return devices

    def create_research_folder(self):
        if not os.path.exists(self.name):
            os.makedirs(self.name)
            for subfolder in ['configs', 'description', 'env', 'experiments']:
                config_path = os.path.join(self.name, subfolder)
                if not os.path.exists(config_path):
                    os.makedirs(config_path)
        else:
            raise ValueError(
                "Research with name '{}' already exists".format(self.name)
            )

    def run(self, name=None, workers=1, branches=1, n_iters=None, devices=None, executor_class=None,
            dump_results=True, parallel=True, executor_target='threads', loglevel='debug', bar=True):
        """ Run research.

        Parameters
        ----------
        name : str, optional
            redefine name of the research (if needed), by default None
        workers : int or list of Config instances, optional
            number of parallel workers, by default 1. If int, number of parallel workers to execute experiments.
            If list of Configs, list of configs for each worker which will be appended to configs from domain. Each
            element corresponds to one worker.
        branches : int or list of Config instances, optional
            number of different branches with different configs with the same root, by default 1. TODO: extend
            If list of Configs, list of configs for each branch which will be appended to configs from domain. Each
            element corresponds to one branch.
        n_iters : int, optional
            number of experiment iterations, by default None, None means that experiment will be executed until
            StopIteration exception.
        devices : str or list, optional
            devices to split between workers and branches, by default None
        executor_class : Executor-inherited class, optional
            executor for experiments, by default None (means that Executor will be used).
        dump_results : bool, optional
            dump results or not, by default True
        parallel : bool, optional
            execute experiments in parallel in separate processes or not, by default True
        executor_target : 'for' or 'threads', optional
            how to execute branches, by default 'threads'
        loglevel : str, optional
            logging level, by default 'debug'

        Returns
        -------
        Research instance

        **How does it work**

        At each iteration all units of the experiment will be executed in the order in which were added.
        If `update_domain` callable is defined, domain will be updated with the corresponding function
        accordingly to `each` parameter of `update_domain`.
        """
        if not parallel:
            if isinstance(workers, int):
                workers = 1
            else:
                workers = [workers[0]]

        self.name = name or self.name
        self.workers = workers
        self.branches = branches
        self.n_iters = n_iters
        self.devices = self.get_devices(devices)
        self.executor_class = executor_class or Executor
        self.dump_results = dump_results
        self.parallel = parallel
        self.executor_target = executor_target
        self.loglevel = loglevel
        self.bar = bar

        if self.dump_results:
            self.create_research_folder()
            self.experiment = self.experiment.dump()
        self.attach_env_meta()

        self.domain.set_iter_params(n_items=self.n_configs, n_reps=self.n_reps, repeat_each=self.repeat_each)

        if self.domain.size is None and (self.domain.update_func is None or self.domain.update_each == 'last'):
            warnings.warn("Research will be infinite because has infinite domain and hasn't domain updating",
                          stacklevel=2)

        self.create_logger()

        self.logger.info("Research is starting")

        n_branches = self.branches if isinstance(self.branches, int) else len(self.branches)
        self.tasks_queue = DynamicQueue(self.domain, self, n_branches)
        self.distributor = Distributor(self.tasks_queue, self)

        self.results = ResearchResults(self.name, self.dump_results)
        self.monitor = ResearchMonitor(self, self.name, bar=self.bar) # process execution signals

        self.monitor.start(self.dump_results)
        self.distributor.run()
        self.monitor.stop()

        return self

    def create_logger(self):
        name = f"{self.name}"
        path = os.path.join(self.name, 'research.log') if self.dump_results else None

        self.logger = create_logger(name, path, self.loglevel)

    def __str__(self):
        spacing = ' ' * 4
        repr = ''

        params = ['name', 'workers', 'branches', 'n_iters', 'devices', 'dump_results',
                  'parallel', 'loglevel', 'executor_target', 'executor_class']
        params_repr = []
        for param in params:
            params_repr += [f"{param}: {getattr(self, param, None)}"]
        params_repr = '\n'.join(params_repr)

        items = {'params': params_repr, 'experiment': str(self.experiment), 'domain': str(self.domain)}
        for name in items:
            repr += f"{name}:\n"
            repr += '\n'.join([spacing + item for item in str(items[name]).split('\n')])
            repr += '\n'

        return repr

class DynamicQueue:
    """ Queue of tasks that can be changed depending on previous results. """
    def __init__(self, domain, research, n_branches):
        self._domain = domain
        self.research = research
        self.n_branches = n_branches

        self.queue = mp.JoinableQueue()

        self.configs_generated = 0
        self.configs_remains = self._domain.size

        self.finished_tasks = 0
        self.tasks_in_queue = 0

    @property
    def domain(self):
        if self._domain.size == 0:
            domain = Domain({'repetition': [None]}) # the value of repetition will be rewritten
            domain.set_iter_params(n_reps=self._domain.n_reps, produced=self._domain.n_produced)
            self._domain = domain
        return self._domain

    def update_domain(self):
        """ Update domain. """
        new_domain = self.domain.update(self.configs_generated, self.research)
        if new_domain is not None:
            self._domain = new_domain
            self.configs_remains = self._domain.size
        return new_domain is not None

    def next_tasks(self, n_tasks=1):
        """ Get next `n_tasks` elements of queue. """
        configs = []
        for i in range(n_tasks):
            branch_tasks = [] # TODO: rename it
            try:
                for _ in range(self.n_branches):
                    branch_tasks.append(next(self.domain))
                configs.append(branch_tasks)
            except StopIteration:
                if len(branch_tasks) > 0:
                    configs.append(branch_tasks)
                break
        for i, executor_configs in enumerate(configs):
            self.put((self.configs_generated + i, executor_configs))

        n_configs = sum([len(item) for item in configs])

        self.configs_generated += n_configs
        self.configs_remains -= n_configs
        self.tasks_in_queue += len(configs)
        self.research.logger.info(f'Get {n_tasks} tasks with {n_configs} configs, remains {self.configs_remains}')

        return n_configs

    def stop_workers(self, n_workers):
        """ Stop all workers by putting `None` task into queue. """
        for _ in range(n_workers):
            self.put(None)

    def join(self):
        self.queue.join()

    def get(self):
        return self.queue.get()

    def put(self, *args, **kwargs):
        return self.queue.put(*args, **kwargs)

    def task_done(self):
        return self.queue.task_done()

    def empty(self):
        return self.queue.empty()

class ResearchMonitor:
    COLUMNS = ['time', 'task_idx', 'id', 'it', 'name', 'status', 'exception', 'worker', 'pid', 'worker_pid',
               'finished', 'withdrawn', 'remains']
    def __init__(self, research, path=None, bar=True):
        self.queue = mp.JoinableQueue()
        self.research = research
        self.path = path
        self.exceptions = mp.Manager().list()
        self.bar = tqdm.tqdm(disable=(not bar)) if isinstance(bar, bool) else bar

        self.finished_experiments = 0
        self.finished_iterations = 0
        self.remained_experiments = None
        self.generated_experiments = 0

        self.current_iterations = {}
        self.n_iters = self.research.n_iters

        self.stop_signal = mp.JoinableQueue()

    @property
    def total(self):
        if self.n_iters:
            return self.finished_iterations + self.n_iters * (self.in_queue + self.remained_experiments)
        else:
            return self.finished_experiments + self.in_queue + self.remained_experiments

    @property
    def in_progress(self):
        return len(self.current_iterations)

    @property
    def in_queue(self):
        return self.generated_experiments - self.finished_experiments

    def send(self, status, experiment=None, worker=None, **kwargs):
        signal = {
            'time': str(datetime.datetime.now()),
            'status': status,
            **kwargs
        }
        if experiment is not None:
            signal = {**signal, **{
                'id': experiment.id,
                'pid': experiment.executor.pid,
            }}
        if worker is not None:
            signal = {**signal, **{
                'worker': worker.index,
                'worker_pid': worker.pid,
            }}
        self.queue.put(signal)
        if 'exception' in signal:
            self.exceptions.append(signal)

    def start_experiment(self, experiment):
        self.send('START_EXP', experiment, experiment.executor.worker)

    def stop_experiment(self, experiment):
        self.send('FINISH_EXP', experiment, experiment.executor.worker, it=experiment.iteration)

    def execute_iteration(self, name, experiment):
        self.send('EXECUTE_IT', experiment, experiment.executor.worker, name=name, it=experiment.iteration)

    def fail_item_execution(self, name, experiment, msg):
        self.send('FAIL_IT', experiment, experiment.executor.worker, name=name, it=experiment.iteration, exception=msg)

    def stop_iteration(self, name, experiment):
        self.send('STOP_IT', experiment, experiment.executor.worker, name=name, it=experiment.iteration)

    def listener(self): #TODO: rename
        signal = self.queue.get()
        filename = os.path.join(self.path, 'monitor.csv')
        with self.bar as progress:
            while signal is not None:
                status = signal.get('status')
                if status == 'TASKS':
                    self.remained_experiments = signal['remains']
                    self.generated_experiments = signal['generated']
                elif status == 'START_EXP':
                    self.current_iterations[signal['id']] = 0
                elif status == 'EXECUTE_IT':
                    self.current_iterations[signal['id']] = signal['it']
                elif status == 'FINISH_EXP':
                    self.current_iterations.pop(signal['id'])
                    self.finished_iterations += signal['it'] + 1
                    self.finished_experiments += 1
                    # self.remained_experiments -= 1

                if status in ['START_EXP', 'EXECUTE_IT', 'FINISH_EXP']:
                    if self.n_iters:
                        progress.n = self.finished_iterations + sum(self.current_iterations.values())
                    else:
                        progress.n = self.finished_experiments + len(self.current_iterations)
                    progress.total = self.total
                    progress.refresh()

                if self.dump:
                    with open(filename, 'a') as f:
                        writer = csv.writer(f)
                        writer.writerow([str(signal.get(column, '')) for column in self.COLUMNS])
                signal = self.queue.get()
        self.stop_signal.put(None)

    def start(self, dump):
        self.dump = dump
        if self.dump:
            filename = os.path.join(self.path, 'monitor.csv')
            if not os.path.exists(filename):
                with open(filename, 'w') as f:
                    writer = csv.writer(f)
                    writer.writerow(self.COLUMNS)
        mp.Process(target=self.listener).start()

    def stop(self):
        self.queue.put(None)
        self.stop_signal.get()

class ResearchResults:
    def __init__(self, name, dump_results):
        self.name = name
        self.dump_results = dump_results
        self.results = mp.Manager().dict()
        self.configs = mp.Manager().dict()

    def put(self, id, results, config):
        self.results[id] = results
        self.configs[id] = config

    def to_df(self, pivot=False, include_config=True, **kwargs):
        df = []
        for experiment_id in self.results:
            experiment_df = []
            for name in self.results[experiment_id]:
                _df = {
                    'id': experiment_id,
                    'iteration': self.results[experiment_id][name].keys()
                }
                if pivot:
                    _df[name] = self.results[experiment_id][name].values()
                else:
                    _df['name'] = name
                    _df['value'] = self.results[experiment_id][name].values()
                experiment_df += [pd.DataFrame(_df)]
            if pivot and len(experiment_df) > 0:
                experiment_df = [functools.reduce(functools.partial(pd.merge, on=['id', 'iteration']), experiment_df)]
            df += experiment_df
        res = pd.concat(df) if len(df) > 0 else pd.DataFrame()
        if include_config and len(res) > 0:
            res = pd.merge(res, self.configs_to_df(**kwargs), how='inner', on='id')
        return res

    def load_iteration_files(self, path, iteration):
        filenames = glob.glob(os.path.join(path, '*'))
        if iteration is None:
            files_to_load = {int(os.path.basename(filename)): filename for filename in filenames}
        else:
            dumped_iteration = np.sort(np.array([int(os.path.basename(filename)) for filename in filenames]))
            files_to_load = dict()
            for it in iteration:
                _it = dumped_iteration[np.argwhere(dumped_iteration >= it)[0, 0]]
                files_to_load[_it] = os.path.join(path, str(_it))
        files_to_load = OrderedDict(sorted(files_to_load.items()))
        results = OrderedDict()
        for filename in files_to_load.values():
            with open(filename, 'rb') as f:
                values = dill.load(f)
                for it in values:
                    if iteration is None or it in iteration:
                        results[it] = values[it]
        return results

    def load(self, experiment_id=None, name=None, iteration=None, config=None, alias=None, domain=None, **kwargs):
        experiment_id = experiment_id if experiment_id is None else to_list(experiment_id)
        name = name if name is None else to_list(name)
        iteration = iteration if iteration is None else to_list(iteration)

        filtered_ids = self.filter_ids_by_configs(config, alias, domain, **kwargs)
        experiment_id = np.intersect1d(experiment_id, filtered_ids) if experiment_id is not None else filtered_ids

        if self.dump_results:
            self.results = OrderedDict()
            for path in glob.glob(os.path.join(self.name, 'experiments', '*', 'results', '*')):
                path = os.path.normpath(path)
                _experiment_id, _, _name = path.split(os.sep)[-3:]
                if experiment_id is None or _experiment_id in experiment_id:
                    if name is None or _name in name:
                        if _experiment_id not in self.results:
                            self.results[_experiment_id] = OrderedDict()
                        experiment_results = self.results[_experiment_id]

                        if _name not in experiment_results:
                            experiment_results[_name] = OrderedDict()
                        name_results = experiment_results[_name]
                        new_values = self.load_iteration_files(path, iteration)
                        experiment_results[_name] = OrderedDict([*name_results.items(), *new_values.items()])

    def configs_to_df(self, use_alias=True, concat_config=False, remove_auxilary=True):
        df = []
        for experiment_id in self.configs:
            config = self.configs[experiment_id]
            if remove_auxilary:
                for key in ['repetition', 'device']:
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
            for config in domain.iterator():
                filtered_ids += self.filter_ids_by_configs(config=config.config())
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
