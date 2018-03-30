#pylint:disable=too-many-instance-attributes
#pylint:disable=too-few-public-methods
#pylint:disable=attribute-defined-outside-init

""" Training of model. """

import os
from copy import copy
from collections import OrderedDict
import pickle

from ..config import Config

class SingleRunning:
    """ Class for training one model repeatedly. """
    def __init__(self):
        self.pipelines = OrderedDict()
        self.config = Config()
        self.results = dict()

    def add_pipeline(self, pipeline, variables=None, name=None, post_run=None, **kwargs):
        """ Add new pipeline to research.
        Parameters
        ----------
        pipeline : dataset.Pipeline
            if preproc is None pipeline must have run action with lazy=True.
        variables : str or list of str or None
            names of pipeline variables to remember at each repetition.
        name : str (default None)
            name of pipeline. If name is None pipeline will have name 'ppl_{index}'
        execute_for : int, list or None
            If -1, pipeline will be executed just at last iteration.
            If other int, pipeline will be excuted for iterations with that step
            If list, pipeline will be excuted for that iterations
            If None, pipeline will executed on each iteration.
        kwargs :
            parameters in pipeline config that depends on the name of the other config. For example,
            if test pipeline imports model from the other pipeline with name 'train' in SingleRunning,
            corresponding parameter in import_model must be C('import_from') and add_pipeline
            must be called with parameter import_from='train'.
        """
        name = name or 'ppl_' + str(len(self.pipelines))
        variables = variables or []
        if not isinstance(variables, list):
            variables = [variables]
        if name in self.pipelines:
            raise ValueError('Pipeline with name {} was alredy existed'.format(name))
        import_config = {key: self.pipelines[value]['ppl'] for key, value in kwargs.items()}

        import_config = Config(import_config)

        self.pipelines[name] = {
            'ppl': pipeline,
            'import_config': import_config,
            'var': variables,
            'post_run': post_run
        }

        self.results[name] = {var: [] for var in variables}
        self.results[name]['iterations'] = []

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

    def add_common_config(self, config):
        """
        Add config that is common for all pipelines.

        Parameters
        ----------
        config : Config or dict
        """
        self.config = Config(config)

    def init(self):
        """
        Add common config to all pipelines.
        """
        for _, pipeline in self.pipelines.items():
            pipeline['ppl'].config += self.config + pipeline['import_config']

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

    def put_result(self, iteration, name):
        """ Put pipeline variable into results. """
        for var in self.pipelines[name]['var']:
            self.results[name][var].append(self.pipelines[name]['ppl'].get_variable(var))
        self.results[name]['iterations'].append(iteration)

    def dump_result(self, name, path):
        """ Dump results. """
        foldername, _ = os.path.split(path)
        if len(foldername) != 0:
            if not os.path.exists(foldername):
                os.makedirs(foldername)
        with open(path, 'wb') as file:
            pickle.dump(self.results[name], file)

    def run_all(self, n_iters):
        """ Run all pipelines. Pipelines will be executed simultaneously in the following sense:
        next_batch is applied successively to each pipeline at each iteration.

        Parameters
        ----------
        n_iters : int
            number of iterations at each repetition
        """
        pipelines = self.pipelines.values()
        for _ in range(n_iters):
            for pipeline in pipelines:
                pipeline['ppl'].next_batch()
        self.results = self.get_results()

    def post_run(self, name):
        """ Run function after run. """
        res = self.pipelines[name]['post_run'](self.pipelines[name]['ppl'])
        for key, value in res.items():
            if key in self.results[name]:
                self.results[name][key].append(value)
            else:
                self.results[name][key] = [value]
