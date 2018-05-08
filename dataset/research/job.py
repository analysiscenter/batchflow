""" Classes Job and Experiment. """

import os
from collections import OrderedDict
from copy import copy
import dill

from .. import Pipeline, Config, inbatch_parallel

class Job:
    """ Contains one job. """
    def __init__(self, config):
        """
        Parameters
        ----------
        config : dict or Config
            config of experiment
        """
        self.experiments = []
        self.config = config

    def init(self, worker_config):
        """ Create experiments. """
        for idx, config in enumerate(self.config['configs']):

            experiment_config = copy(self.config)

            if isinstance(experiment_config['branches'], list):
                branch_config = experiment_config['branches'][idx]
            else:
                branch_config = Config()

            experiment_config['configs'] = experiment_config['configs'][idx]
            experiment_config['repetition'] = experiment_config['repetition'][idx]
            experiment_config['idx'] = idx

            experiment = Experiment({
                **experiment_config,
                'worker_config': worker_config,
                'branch_config': branch_config
            })

            for name, pipeline in self.config['pipelines'].items():
                experiment.add_pipeline(pipeline, config, name)

            for name, function in self.config['functions'].items():
                experiment.add_function(function, name)

            self.experiments.append(experiment)

    def parallel_execute_for(self, name):
        """ Parallel execution of pipeline 'name'. """
        batch = self.config['pipelines'][name]['root'].next_batch()
        self._parallel_run(batch, name)


    @inbatch_parallel(init='_parallel_init')
    def _parallel_run(self, item, batch, name):
        item.execute_for(batch, name)

    def _parallel_init(self, batch, name):
        _ = batch, name
        return self.experiments

    def get_description(self):
        """ Get description of job. """
        if isinstance(self.config['branches'], list):
            description = '\n'.join([str({**config.alias(), **_config})
                                     for config, _config in zip(self.config['configs'], self.config['branches'])])
        else:
            description = '\n'.join([str(config.alias()) for config in self.config['configs']])
        return description

    def put_pipeline_result(self, iteration, name):
        """ Add values of pipeline variables to results. """
        for experiment in self.experiments:
            experiment.put_pipeline_result(iteration, name)

    def dump_all(self):
        """ Dump all results. """
        for name in self.config['pipelines']:
            for item in self.experiments:
                item.dump_pipeline_result(name, name)
        for name, function in self.config['functions'].items():
            if len(function['returns']) > 0:
                for item in self.experiments:
                    item.dump_function_result(name, name)


class Experiment:
    """ Corresponds to one config. """
    def __init__(self, config):
        self.config = config
        self.pipelines = OrderedDict()
        self.functions = OrderedDict()
        self.pipeline_results = dict()
        self.function_results = dict()
        self.config['path'] = self.get_path()
        if not os.path.exists(self.config['path']):
            os.makedirs(self.config['path'])

    def add_pipeline(self, pipeline, config, name):
        """ Add new pipeline to research. """
        name = name or 'ppl_' + str(len(self.pipelines))
        pipeline['var'] = pipeline['var'] or []
        if not isinstance(pipeline['var'], list):
            pipeline['var'] = [pipeline['var']]
        if name in self.pipelines:
            raise ValueError('Pipeline with name {} was alredy existed'.format(name))
        import_config = {key: self.pipelines[value]['ppl'] for key, value in pipeline['kwargs'].items()}

        pipeline_config = Config(config.config() + self.config['worker_config']
            + self.config['branch_config'] + import_config)

        pipeline['execute_for'] = self.get_iterations(pipeline['execute_for'], self.config['n_iters'])
        if pipeline['dump_for'] is not None:
            pipeline['dump_for'] = self.get_iterations(pipeline['dump_for'], self.config['n_iters'])
        else:
            pipeline['dump_for'] = []


        self.pipelines[name] = copy(pipeline)
        self.pipelines[name]['ppl'] = self.pipelines[name]['ppl'] + Pipeline()
        self.pipelines[name]['ppl'].set_config(pipeline_config)

        self.pipeline_results[name] = {var: [] for var in pipeline['var']}
        self.pipeline_results[name]['iterations'] = []

    def add_function(self, function, name):
        """ Add function to research. """
        function['execute_for'] = self.get_iterations(function['execute_for'], self.config['n_iters'])
        if function['dump_for'] is not None:
            function['dump_for'] = self.get_iterations(function['dump_for'], self.config['n_iters'])
        else:
            function['dump_for'] = []

        function['returns'] = function['returns'] or []
        if not isinstance(function['returns'], list):
            function['returns'] = [function['returns']]

        self.functions[name] = copy(function)
        self.functions[name]['kwargs'] = {
            key: self.get_pipeline(value)['ppl']
            for key, value in function['kwargs'].items()
        }
        if len(function['returns']) > 0:
            self.function_results[name] = {var: [] for var in function['returns']}
            self.function_results[name]['iterations'] = []

    def call_function(self, iteration, name):
        """ Call function 'name'. """
        res = self.functions[name]['function'](self, iteration, **self.functions[name]['kwargs'])
        if not isinstance(res, list):
            res = [res]
        if len(self.functions[name]['returns']) > 0:
            for var, value in zip(self.functions[name]['returns'], res):
                self.function_results[name][var].append(value)
            self.function_results[name]['iterations'].append(iteration)

    def execute_for(self, batch, name):
        """ Execute pipeline 'name' for batch. """
        self.pipelines[name]['ppl'].execute_for(batch)

    def next_batch(self, name):
        """
        Get next batch from pipleine.

        Parameters
        ----------
        name : str
            pipeline name
        """
        self.pipelines[name]['ppl'].next_batch()

    def run(self, name, reset=True):
        """ Run pipelines till the end. """
        if reset:
            self.pipelines[name]['ppl'].reset_iter()
        self.pipelines[name]['ppl'].run()

    def post_run(self, name):
        """ Run function after run. """
        res = self.pipelines[name]['post_run'](self.pipelines[name]['ppl'])
        for key, value in res.items():
            if key in self.results[name]:
                self.results[name][key].append(value)
            else:
                self.results[name][key] = [value]

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

    def _variable_len(self, name, variable):
        if name in self.pipelines:
            return len(self.pipelines[name]['ppl'].get_variable(variable))
        else:
            return None

    def get_pipeline(self, name):
        """
        Parameters
        ----------
        name : str
        """
        return self.pipelines[name]

    def run_on_batch(self, batch, name):
        """
        Run pipeline on prepared batch.

        Parameters
        ----------
        batch : dataset.Batch

        name : str
            pipeline name
        """
        self.pipelines[name]['ppl'].execute_for(batch)

    def put_pipeline_result(self, iteration, name):
        """ Put values of pipeline variables to results. """
        if len(self.pipelines[name]['var']) > 0:
            for var in self.pipelines[name]['var']:
                self.pipeline_results[name][var].append(self.pipelines[name]['ppl'].get_variable(var))
            self.pipeline_results[name]['iterations'].append(iteration)

    def get_path(self):
        """ Get path to folder where results will be saved. """
        return os.path.join(
            self.config['name'],
            'results',
            self.config['configs'].alias(as_string=True),
            str(self.config['repetition'])
        )

    def dump_pipeline_result(self, name, filename):
        """ Dump pipeline results. """
        if len(self.pipelines[name]['var']) > 0:
            foldername = self.config['path']
            path = os.path.join(foldername, filename)
            with open(path, 'wb') as file:
                dill.dump(self.pipeline_results[name], file)

    def dump_function_result(self, name, filename):
        """ Dump function results. """
        if len(self.functions[name]['returns']) > 0:
            foldername = self.config['path']
            path = os.path.join(foldername, filename)
            with open(path, 'wb') as file:
                dill.dump(self.function_results[name], file)
