""" Classes Job and Experiment. """

import time
from collections import OrderedDict
import random

from ..named_expr import eval_expr
from .. import inbatch_parallel, make_seed_sequence

class Job:
    """ Contains one job. """
    def __init__(self, executable_units, n_iters, configs, branches, research_path):
        """
        Parameters
        ----------
        config : dict or Config
            config of experiment
        """
        self.experiments = []
        self.executable_units = executable_units
        self.n_iters = n_iters
        self.configs = configs
        self.branches = branches
        self.research_path = research_path
        self.worker_config = {}
        self.ids = [str(random.getrandbits(32)) for _ in self.configs]

        self.exceptions = []
        self.stopped = []
        self.last_update_time = None

        self.random_seed = make_seed_sequence()

    def init(self, worker_config, device_configs, last_update_time):
        """ Create experiments. """
        self.worker_config = worker_config
        self.last_update_time = last_update_time
        for index, (config, additional_config) in enumerate(self.configs):
            if isinstance(self.branches, list):
                branch_config = self.branches[index]
            else:
                branch_config = dict()
            units = OrderedDict()
            for name, unit in self.executable_units.items():
                unit = unit.get_copy()
                unit.set_shared_value(last_update_time)
                unit.reset('iter')
                if unit.pipeline is not None:
                    kwargs_config = eval_expr(unit.kwargs, job=self, experiment=units)
                    unit.set_dataset()
                else:
                    kwargs_config = dict()
                unit.set_config(config, additional_config,
                                {**branch_config, **device_configs[index]}, worker_config, kwargs_config)
                unit.set_research_path(self.research_path)
                unit.dump_config()
                unit.index = index
                unit.create_folder()
                units[name] = unit


            self.experiments.append(units)
            self.exceptions.append(None)
        self.clear_stopped_list()

    def clear_stopped_list(self):
        """ Clear list of stopped experiments for the current iteration """
        self.stopped = [False for _ in range(len(self.experiments))]

    def get_description(self):
        """ Get description of job. """
        if isinstance(self.branches, list):
            description = '\n'.join([str({**config[0].alias(), **_config, **self.worker_config})
                                     for config, _config in zip(self.configs, self.branches)])
        else:
            description = '\n'.join([str({**config[0].alias(), **self.worker_config})
                                     for config in self.configs])
        return description

    def parallel_execute_for(self, iteration, name, actions):
        """ Parallel execution of pipeline 'name' """
        unit = self.experiments[0][name]
        run = [action['run'] for action in actions if action is not None][0]
        if run:
            unit.reset_root_iter()
            while True:
                try:
                    batch = unit.next_batch_root()
                    exceptions = self._parallel_run(iteration, name, batch, actions)
                except StopIteration:
                    break
        else:
            try:
                batch = unit.next_batch_root()
            except StopIteration as e:
                exceptions = [e] * len(self.experiments)
            else:
                exceptions = self._parallel_run(iteration, name, batch, actions)
        self.put_all_results(iteration, name, actions)
        self.last_update_time.value = time.time()
        return exceptions

    def update_exceptions(self, exceptions):
        """ Update exceptions with new from current iteration """
        for i, exception in enumerate(exceptions):
            if exception is not None:
                self.exceptions[i] = exception

    @inbatch_parallel(init='_parallel_init_run', post='_parallel_post')
    def _parallel_run(self, item, execute, iteration, name, batch, actions):
        _ = name, actions, iteration
        if execute is not None:
            item.execute_for(batch)
        return None

    def _parallel_init_run(self, iteration, name, batch, actions):
        _ = iteration, batch
        #to_run = self._experiments_to_run(iteration, name)
        return [[experiment[name], execute] for experiment, execute in zip(self.experiments, actions)]

    @inbatch_parallel(init='_parallel_init_call', post='_parallel_post')
    def parallel_call(self, experiment, execute, iteration, name, actions):
        """ Parallel call of the unit 'name' """
        _ = actions
        if execute is not None:
            experiment[name](job=self, iteration=iteration, experiment=experiment)

    def _parallel_init_call(self, iteration, name, actions):
        _ = iteration, name
        return [[experiment, execute] for experiment, execute in zip(self.experiments, actions)]

    def _parallel_post(self, results, *args, **kwargs):
        _ = args, kwargs
        self.last_update_time.value = time.time()
        return results

    def call_on_root(self, iteration, unit_name):
        """ Callable on root """
        try:
            unit = self.executable_units[unit_name]
            unit(job=self, iteration=iteration, experiment=self.experiments)
            return [None] * len(self.experiments)
        except Exception as e: #pylint:disable=broad-except
            return [e] * len(self.experiments)

    def put_all_results(self, iteration, name, actions):
        """ Add values of pipeline variables to results """
        for experiment, execute in zip(self.experiments, actions):
            if execute is not None:
                experiment[name].put_result(iteration)

    def get_actions(self, iteration, name, action='execute'):
        """ Experiments that should be executed """
        res = []
        for idx, experiment in enumerate(self.experiments):
            if experiment[name].action_iteration(iteration, self.n_iters, action) and self.exceptions[idx] is None:
                res.append(experiment[name].action)
            elif (self.stopped[idx]) and 'last' in getattr(experiment[name], action):
                res.append(experiment[name].action)
            else:
                res.append(None)
        return res

    def all_stopped(self):
        """ Does all experiments finished """
        res = True
        for exception in self.exceptions:
            res = isinstance(exception, StopIteration)
        return res

    def alive_experiments(self):
        """ Get number of alive experiments """
        return len([item for item in self.exceptions if item is None])
