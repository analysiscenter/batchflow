""" Classes Job and Experiment. """

import os
from collections import OrderedDict
from copy import copy
import dill

from .. import Pipeline, Config, inbatch_parallel

class Job:
    """ Contains one job. """
    def __init__(self, executable_units, n_iters, repetition, configs, branches, name):
        """
        Parameters
        ----------
        config : dict or Config
            config of experiment
        """
        self.experiments = []
        # self.config = config

        self.executable_units = executable_units
        self.n_iters = n_iters
        self.configs = configs
        self.repetition = repetition
        self.branches = branches
        self.name = name

    def init(self, worker_config):
        """ Create experiments. """
        self.worker_config = worker_config

        for unit in self.executable_units.values():
            unit.execute_for = self.get_iterations(unit.execute_for, self.n_iters)
            unit.dump_for = self.get_iterations(unit.dump_for, self.n_iters)

        for index, config in enumerate(self.configs):
            if isinstance(self.branches, list):
                branch_config = self.branches[index]
            else:
                branch_config = dict()
            units = OrderedDict()
            for name, unit in self.executable_units.items():
                unit = unit.get_copy()
                if unit.pipeline:
                    import_config = {key: units[value].pipeline for key, value in unit.kwargs.items()}
                else:
                    import_config = dict()
                unit.set_config(config, branch_config, worker_config, import_config)
                unit.repetition = self.repetition[index]
                unit.index = index
                unit.create_folder(self.name)
                units[name] = unit

            self.experiments.append(units)

    def get_iterations(self, execute_for, n_iters=None):
        """ Get indices of iterations from execute_for. """
        if n_iters is not None:
            if isinstance(execute_for, int):
                if execute_for == -1:
                    execute_for = [n_iters - 1]
                else:
                    execute_for = list(range(-1, n_iters, execute_for))
            elif execute_for is None:
                execute_for = list(range(n_iters))
        return execute_for

    def get_description(self):
        """ Get description of job. """
        if isinstance(self.branches, list):
            description = '\n'.join([str({**config.alias(), **_config, **self.worker_config})
                                     for config, _config in zip(self.configs, self.branches)])
        else:
            description = '\n'.join([str({**config.alias(), **self.worker_config})
                                     for config in self.configs])
        return description

    def parallel_execute_for(self, name):
        """ Parallel execution of pipeline 'name'. """
        batch = self.executable_units[name].next_batch_root()
        self._parallel_run(batch, name)


    @inbatch_parallel(init='_parallel_init')
    def _parallel_run(self, item, batch, name):
        _ = name
        item.execute_for(batch)

    def _parallel_init(self, batch, name):
        _ = batch, name
        return [experiment[name] for experiment in self.experiments]
    

    def put_results(self, iteration, name, result=None):
        """ Add values of pipeline variables to results. """
        for experiment in self.experiments:
            unit[name].put_results(iteration, result) 
