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
        self.results = None

    def add_pipeline(self, pipeline, variables=None, config=None, name=None, execute_for=None, **kwargs):
        """ Add new pipeline to research.
        Parameters
        ----------
        pipeline : dataset.Pipeline
            if preproc is None pipeline must have run action with lazy=True.
        variables : str or list of str or None
            names of pipeline variables to remember at each repetition.
        config : Config or dict (default None)
            pipeline config
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

        config = Config(config)
        import_config = Config(import_config)

        self.pipelines[name] = {
            'ppl': pipeline,
            'cfg': config + import_config,
            'var': variables,
            'execute_for': execute_for
        }

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

    def get_results(self):
        """ Get values of variables from pipelines.
        Returns
        -------
        Results

        If some pipeline was added without variables it will not be included into results.
        """
        return Results(self.pipelines)

    def init(self):
        """
        Add common config to all pipelines.
        """
        for _, pipeline in self.pipelines.items():
            pipeline['ppl'].set_config(pipeline['cfg'] + self.config)

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
        return self.pipelines[name]['ppl'].next_batch()

    def run(self, n_iters):
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

    def save_results(self, save_to):
        """ Pickle results to file.

        Parameters
        ----------
        save_to : str
        """
        self.results = self.get_results()
        foldername, _ = os.path.split(save_to)
        if len(foldername) != 0:
            if not os.path.exists(foldername):
                os.makedirs(foldername)
        with open(save_to, 'wb') as file:
            pickle.dump(self.results, file)

    @classmethod
    def get_iterations(cls, execute_for, n_iters=None):
        """ Get indices of iterations from execute_for. """
        if n_iters is not None:
            if isinstance(execute_for, int):
                if execute_for == -1:
                    execute_for = [n_iters - 1]
                else:
                    execute_for = list(range(0, n_iters, execute_for))
            elif execute_for is None:
                execute_for = list(range(n_iters))
        return execute_for


class Results(Config):
    """ Results of single experiment. """
    def __init__(self, pipelines, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._add_results(pipelines)

    def _add_results(self, pipelines):
        for name, pipeline in pipelines.items():
            if len(pipeline['var']) != 0:
                variables = {
                    variable: copy(pipeline['ppl'].get_variable(variable)) for variable in pipeline['var']
                }
                self[name] = {
                    'iterations': pipeline['execute_for'],
                    **variables
                }
