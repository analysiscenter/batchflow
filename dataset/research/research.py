#pylint:disable=too-many-instance-attributes
#pylint:disable=too-few-public-methods
#pylint:disable=attribute-defined-outside-init
#pylint:disable=bare-except
#pylint:disable=no-value-for-parameter

""" Class Research and auxiliary classes for multiple experiments. """

import os
from copy import copy
from collections import OrderedDict
from math import ceil
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

    def add_pipeline(self, root_pipeline, branch_pipeline=None, variables=None, config=None, name=None,
                     execute_for=None, run=False, **kwargs):
        """ Add new pipeline to research.

        Parameters
        ----------
        root_pipeline : dataset.Pipeline
            root_pipeline must have run action with lazy=True and n_epochs=None. If branch_pipeline=None then
            may contain parameters that can be defined by grid.
        branch_pipeline : dataset.Pipeline or None
            if not None, for resulting batch from root_pipeline branch_pipeline.execute_for(batch) will be called.
            May contain parameters that can be defined by grid.
        variables : str, list of str or None
            names of pipeline variables to save after each repetition. All of them must be defined in pipeline,
            not in preproc.
        preproc : dataset.Pipeline or None
            if preproc is not None it must have run action with lazy=True and n_epochs=None. For resulting batch
            pipeline.execute_for(batch) will be called. Notice that pipeline must not change batch from preproc.
        name : str (default None)
            pipeline name. If name is None, pipeline will have name 'ppl_{index}'
        execute_for : int, list of ints or None
            If -1, pipeline will be executed just at last iteration.
            If positive int, pipeline will be excuted for iterations with that step
            If list of ints, pipeline will be excuted for that iterations
            If None, pipeline will executed at each iteration.
        run : bool (default False)
            if False then .next_batch() will be applied to pipeline, else .run().
        kwargs :
            parameters in pipeline config that depends on the names of the other pipeline. For example,
            if test pipeline imports model from the other pipeline with name 'train' in Researcn,
            corresponding parameter in import_model must be C('import_from') and add_pipeline
            must be called with parameter import_from='train'.

        **How to define changing parameters*

        All parameters in root_pipeline or branch_pipeline that are defined in grid should be defined
        as C('parameter_name'). Corresponding parameter in grid must have the same 'parameter_name'.
        """
        name = name or 'ppl_' + str(len(self.pipelines))
        config = config or Config()
        variables = variables or []

        if not isinstance(variables, list):
            variables = [variables]
        if name in self.pipelines:
            raise ValueError('Pipeline with name {} was alredy existed'.format(name))

        if branch_pipeline is None:
            pipeline = root_pipeline
            preproc = None
        else:
            pipeline = branch_pipeline
            preproc = root_pipeline

        self.pipelines[name] = {
            'ppl': pipeline,
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

    def _create_tasks(self, n_reps, n_iters, n_branches, name):
        """ Create Tasks instance with tasks to run. Each task is one repetition for each config
        from grid_config.

        Parameters
        ----------
        n_reps : int

        n_iters : int

        n_branches : int

        name : str
            name of research.
        """
        if isinstance(n_branches, int):
            n_models = n_branches
        elif n_branches is None:
            n_models = 1
        else:
            n_models = len(n_branches)

        configs_with_repetitions = [(idx, configs)
                                    for idx in range(n_reps)
                                    for configs in self.grid_config.gen_configs()]

        configs_chunks = self._chunks(configs_with_repetitions, n_models)

        self.tasks = (
            {'pipelines': self.pipelines,
             'n_iters': n_iters,
             'configs': list(zip(*chunk))[1],
             'repetition': list(zip(*chunk))[0],
             'n_branches': n_branches,
             'name': name
            }
            for chunk in configs_chunks
        )
        self.n_tasks = ceil(len(configs_with_repetitions) / n_models)
        self.tasks = Tasks(self.tasks)

    def _chunks(self, array, size):
        """ Divide array into chunks of the fixed size.

        Parameters
        ----------
        array : list or np.ndarray

        size : int
            chunk size.
        """
        for i in range(0, len(array), size):
            yield array[i:i + size]

    def run(self, n_reps, n_iters, n_workers=1, n_branches=1, name=None, progress_bar=False, save_model=False):
        """ Run research.

        Parameters
        ----------
        n_reps : int
            number of repetitions with each combination of parameters from grid_config.
        n_iters: int
            number of iterations for each configurations of each pipeline.
        n_workers : int (default 1), list of instances of Worker or list of dicts (Configs).
            Workers (processes) to run tasks in parallel.
            If int - number of workers to run pipelines or workers that will run them. By default,
                PipelineWorker will be used.
            If list of instances of Worker - workers to run tasks.
            If list of dicts (Configs) - list of additional configs which will be appended to configs from tasks.
                Each element corresponds to worker.
        n_branches: int or list of dicts (Configs)
            Is using if preproc is not None. Number of different configs which will use the same batch
            from preproc pipeline. Pipelines will be executed in different threads.
            If int - number of pipelines with different configs that will use the same prepared batch
                from preproc.
            If list of dicts (Configs) - list of dicts with additional configs to each pipeline.

            For example, if there are 2 GPUs, we can define parameter 'device' in model config as C('device')
            and define n_branches as [{'device': 0}, {'device': 1}].
        name : str or None
            name folder to save research. By default is 'research'.
        progress_bar : bool (default False)
            add tqdm progress bar
        save_model : bool
            save or not the model with name 'model' at the first repetition from 'train' pipeline.
            If n_workers is not int there is no difference between True and False.

        At each iteration all add pipelines will be runned with some config from grid.
        """
        self.n_reps = n_reps
        self.n_iters = n_iters
        self.n_workers = n_workers
        self.n_branches = n_branches

        self.name = self._does_exist(name)

        # dump information about research
        self.save()

        self._create_tasks(n_reps, n_iters, n_branches, self.name)

        if isinstance(n_workers, int) or isinstance(n_workers[0], (dict, Config)):
            if save_model:
                worker = SavingWorker # worker that saves model at first repetition
            else:
                worker = PipelineWorker
        else:
            worker = None
        distr = Distributor(n_workers, worker)
        distr.run(self.tasks, dirname=self.name, n_tasks=self.n_tasks, progress_bar=progress_bar)
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
        description['pipelines'] = _pipelines
        return description

    @classmethod
    def load(cls, name):
        """ Load description of the research from name/description. """
        with open(os.path.join(name, 'description'), 'rb') as file:
            return dill.load(file)
