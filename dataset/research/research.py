#pylint:disable=too-many-instance-attributes
#pylint:disable=too-few-public-methods
#pylint:disable=attribute-defined-outside-init
#pylint:disable=bare-except
#pylint:disable=no-value-for-parameter

""" Class Research and auxiliary classes for multiple experiments. """

import os
from copy import copy
from collections import OrderedDict
import json
import dill

from ..config import Config
from .distributor import Tasks, Distributor
from .workers import PipelineWorker, SavingWorker
from .grid import Grid

class Research:
    """ Class Research for multiple experiments with pipelines. """
    def __init__(self):
        self.pipelines = OrderedDict()

    def add_pipeline(self, pipeline, variables, preproc=None, config=None, name=None,
                     execute_for=None, run=False, **kwargs):
        """ Add new pipeline to research.

        Parameters
        ----------
        pipeline : dataset.Pipeline
            if preproc is None, pipeline must have run action with lazy=True. All parameters that are
            defined in grid should be defined as C('parameter_name'). Corresponding parameter in grid
            must have the same 'parameter_name'
        variables : str or list of str
            names of pipeline variables to remember at each repetition. All of them must be defined in pipeline,
            not in preproc.
        preproc : dataset.Pipeline or None
            if preproc is not None it must have run action with lazy=True. For resulting batch
            pipeline.execute_for(batch) will be called.
        config : Config or dict (default None)
            pipeline config with parameters that doesn't change between experiments.
        name : str (default None)
            pipeline name. If name is None, pipeline will have name 'ppl_{index}'
        execute_for : int, list of ints or None
            If -1, pipeline will be executed just at last iteration.
            If positive int, pipeline will be excuted for iterations with that step
            If list of ints, pipeline will be excuted for that iterations
            If None, pipeline will executed at each iteration.
        kwargs :
            parameters in pipeline config that depends on the names of the other pipeline. For example,
            if test pipeline imports model from the other pipeline with name 'train' in Researcn,
            corresponding parameter in import_model must be C('import_from') and add_pipeline
            must be called with parameter import_from='train'.
        """
        name = name or 'ppl_' + str(len(self.pipelines))
        config = config or Config()
        variables = variables or []

        if not isinstance(variables, list):
            variables = [variables]
        if name in self.pipelines:
            raise ValueError('Pipeline with name {} was alredy existed'.format(name))

        self.pipelines[name] = {
            'ppl': pipeline,
            'cfg': config,
            'var': variables,
            'execute_for': execute_for,
            'preproc': preproc,
            'kwargs': kwargs,
            'run': run
        }

        return self

    def add_grid_config(self, grid_config):
        """ Add grid of pipeline parameters.

        Parameters
        ----------
        grid_config : dict, Grid or Option
            if dict it should have items parameter_name: list of values.
        """
        self.grid_config = Grid(grid_config)
        return self

    def _create_tasks(self, n_reps, n_iters, n_groups, name):
        if isinstance(n_groups, int):
            n_models = n_groups
        elif n_groups is None:
            n_models = 1
        else:
            n_models = len(n_groups)

        configs_with_repetitions = [(idx, configs)
                                    for idx in range(n_reps)
                                    for configs in self.grid_config.gen_configs()]

        configs_chunks = self._chunks(configs_with_repetitions, n_models)

        self.tasks = (
            {'pipelines': self.pipelines,
             'n_iters': n_iters,
             'configs': list(zip(*chunk))[1],
             'repetition': list(zip(*chunk))[0],
             'n_groups': n_groups,
             'name': name
            }
            for chunk in configs_chunks
        )
        self.tasks = Tasks(self.tasks)

    def _chunks(self, array, size):
        """ Divide array into chunks of the fixed size. """
        for i in range(0, len(array), size):
            yield array[i:i + size]

    def run(self, n_reps, n_iters, n_workers=1, n_groups=1, name=None, save_model=False):
        """ Run research.

        Parameters
        ----------
        n_reps : int
            number of repetitions with each combination of parameters
        n_iters: int
            number of iterations for each configurations of each pipeline.
        n_workers : int (default 1) or list of Workers
            If int - number of workers to run pipelines or workers that will run them. By default,
            PipelineWorker will be used.
            If list - instances of Worker class.
        n_groups: int or list of dicts
            If int - number of pipelines with different configs that will use the same prepared batch
            from preproc. If n_groups - list of dicts with additional configs to each pipeline.
            For example, if there are 2 GPUs, we can define parameter 'device' in model config as C('device')
            and define n_groups as [{'device': 0}, {'device': 1}].
        name : str or None
            name folder to save research. By default is 'research'.
        save_model : bool
            save or not the model 'model' at the first repetition from 'train' pipeline.
            If n_workers is not int there is no difference between True and False.

        At each iteration all add pipelines will be runned with some config from grid.
        """
        self.n_reps = n_reps
        self.n_iters = n_iters
        self.n_workers = n_workers
        self.n_groups = n_groups

        self.name = self._does_exist(name)

        # dump information about research
        self.save()

        self._create_tasks(n_reps, n_iters, n_groups, self.name)

        if isinstance(n_workers, int) or isinstance(n_workers[0], (dict, Config)):
            if save_model:
                worker = SavingWorker # worker that saves model at first repetition
            else:
                worker = PipelineWorker
        else:
            worker = None
        distr = Distributor(n_workers, worker)
        distr.run(self.tasks, dirname=self.name)
        return self

    def _does_exist(self, name):
        name = name or 'research'
        if not os.path.exists(name):
            dirname = name
        else:
            i = 1
            while os.path.exists(name + '_' + str(i)):
                i += 1
            dirname = name + '_' + str(i)
        os.makedirs(dirname)
        return dirname

    def save(self):
        """ Save description of the research to folder name/description. """
        with open(os.path.join(self.name, 'description'), 'wb') as file:
            dill.dump(self, file)
        with open(os.path.join(self.name, 'description_research.json'), 'w') as file:
            file.write(json.dumps(self._json(), default=self._set_default_json))
        with open(os.path.join(self.name, 'description_alias.json'), 'w') as file:
            file.write(json.dumps(self.grid_config.description(), default=self._set_default_json))

    def _set_default_json(self, obj):
        try:
            json.dumps(obj)
        except TypeError:
            return str(obj)

    def _json(self):
        description = copy(self.__dict__)
        description['grid_config'] = self.grid_config.value()
        _pipelines = dict()
        for name, pipeline in self.pipelines.items():
            _pipelines[name] = copy(pipeline)
            del _pipelines[name]['ppl']
            _pipelines[name]['cfg'] = _pipelines[name]['cfg'].flatten()
        description['pipelines'] = _pipelines
        return description

    @classmethod
    def load(cls, name):
        """ Load description of the research from name/description. """
        with open(os.path.join(name, 'description'), 'rb') as file:
            return dill.load(file)
