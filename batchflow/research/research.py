import os
import datetime
import csv
import itertools
import subprocess
import re
import dill
import multiprocess as mp
import glob
import tqdm
from collections import OrderedDict

import warnings

from .domain import Domain
from .distributor import Distributor, DynamicQueue
from .experiment import Experiment, Executor
from .results import ResearchResults
from .utils import create_logger

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
        self._results = None

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
        os.makedirs(self.name)
        for subfolder in ['configs', 'description', 'env', 'experiments']:
            config_path = os.path.join(self.name, subfolder)
            if not os.path.exists(config_path):
                os.makedirs(config_path)


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

        if dump_results and os.path.exists(self.name):
            raise ValueError(f"Research with name '{self.name}' already exists")

        self.domain.set_iter_params(n_items=self.n_configs, n_reps=self.n_reps, repeat_each=self.repeat_each)

        if self.domain.size is None and (self.domain.update_func is None or self.domain.update_each == 'last'):
            warnings.warn("Research will be infinite because has infinite domain and hasn't domain updating",
                          stacklevel=2)

        if self.dump_results:
            self.create_research_folder()
            self.experiment = self.experiment.dump() # add final dump of experiment results
            self.dump_research()
        self.attach_env_meta()

        self.create_logger()

        self.logger.info("Research is starting")

        n_branches = self.branches if isinstance(self.branches, int) else len(self.branches)
        self.tasks_queue = DynamicQueue(self.domain, self, n_branches)
        self.distributor = Distributor(self.tasks_queue, self)

        self.monitor = ResearchMonitor(self, self.name, bar=self.bar) # process execution signals
        self.results = ResearchResults(self.name, self.dump_results)

        self.monitor.start(self.dump_results)
        self.distributor.run()
        self.monitor.stop()

        return self

    def create_logger(self):
        name = f"{self.name}"
        path = os.path.join(self.name, 'research.log') if self.dump_results else None

        self.logger = create_logger(name, path, self.loglevel)

    def dump_research(self):
        with open(os.path.join(self.name, 'research.dill'), 'wb') as f:
            dill.dump(self, f)

    @classmethod
    def load(cls, name):
        with open(os.path.join(name, 'research.dill'), 'rb') as f:
            research = dill.load(f)
        research.results = ResearchResults(research.name, research.dump_results)
        return research

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
