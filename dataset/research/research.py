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
import warnings
import dill

from .. import Config
from .distributor import Distributor
from .workers import PipelineWorker
from .grid import Grid

class Research:
    """ Class Research for multiple parallel experiments with pipelines. """
    def __init__(self):
        self.pipelines = OrderedDict()
        self.functions = OrderedDict()

    def pipeline(self, root_pipeline, branch_pipeline=None, variables=None, name=None,
                 execute_for=None, dump_for=None, run=False, **kwargs):
        """ Add new pipeline to research.

        Parameters
        ----------
        root_pipeline : dataset.Pipeline
            root_pipeline must have run action with lazy=True. If branch_pipeline is None then root_pipeline
            may contain parameters that can be defined by grid.
        branch_pipeline : dataset.Pipeline or None
            if not None, for resulting batch from root_pipeline branch_pipeline.execute_for(batch) will be called.
            May contain parameters that can be defined by grid.
        variables : str, list of str or None
            names of pipeline variables to save after each repetition. All of them must be defined in root_pipeline
            if branch_pipeline is None or in branch_pipeline if branch_pipeline is not None.
        name : str (default None)
            pipeline name inside research. If name is None, pipeline will have name 'ppl_{index}'
        execute_for : int, list of ints or None
            If -1, pipeline will be executed just at last iteration.
            If positive int, pipeline will be excuted for iterations with that step
            If list of ints, pipeline will be excuted for that iterations
            If None, pipeline will executed at each iteration.
        dump_for : int, list of ints or None
            iteration when results will be dumped. Similar to execute_for
            If None, pipeline results will not be dumped.
        run : bool (default False)
            if False then .next_batch() will be applied to pipeline, else .run() and then reset_iter().
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
        variables = variables or []

        if not isinstance(variables, list):
            variables = [variables]
        if name in self.pipelines:
            raise ValueError('Pipeline with name {} was alredy existed'.format(name))

        if branch_pipeline is None:
            pipeline = root_pipeline
            root = None
        else:
            pipeline = branch_pipeline
            root = root_pipeline

        self.pipelines[name] = {
            'ppl': pipeline,
            'root': root,
            'var': variables,
            'execute_for': execute_for,
            'dump_for': dump_for,
            'run': run,
            'kwargs': kwargs,
        }

        return self

    def function(self, function, name, execute_for=None, dump_for=None, returns=None, *args, **kwargs):
        """ Add function to research.

        Parameters
        ----------
        function : callable
            callable object with following parameters:
                experiment : Experiment
                    class which contains config and results of the experiment
                iteration : int
                    iteration when function is called
                **args, **kwargs
        name : str (default None)
            function name inside research. If name is None, pipeline will have name 'func_{index}'
        execute_for : int, list of ints or None
            If -1, function will be called just at last iteration.
            If positive int, function will be called for iterations with that step
            If list of ints, function will be called for that iterations
            If None, function will called at each iteration.
        dump_for : int, list of ints or None
            iteration when results will be dumped. Similar to execute_for
            If None, function results will not be dumped.
        returns : str, list of str or None
            names for function results.
        """
        name = name or 'func_' + str(len(self.functions))
        self.functions[name] = {
            'function': function,
            'execute_for': execute_for,
            'dump_for': dump_for,
            'returns': returns,
            'args': args,
            'kwargs': kwargs
        }
        return self

    def grid(self, grid_config):
        """ Add grid of pipeline parameters.

        Parameters
        ----------
        grid_config : dict, Grid or Option
            if dict it should have items parameter_name: list of values.

        Configs from that grid will be generated and then substitute into pipelines.
        """
        self.grid_config = Grid(grid_config)
        return self

    def _create_jobs(self, n_reps, n_iters, branches, name):
        """ Create generator of jobs. If branches=1 or len(branches)=1 then each job is one repetition
        for each config from grid_config. Else each job contains several pairs (repetition, config).

        Parameters
        ----------
        n_reps : int

        n_iters : int

        branches : int or list of Configs
            if int, branches is a number of branches for one root
            if list then each element is additional Configs for pipelines
        name : str
            name of research.
        """
        if isinstance(branches, int):
            n_models = branches
        elif branches is None:
            n_models = 1
        else:
            n_models = len(branches)

        configs_with_repetitions = [(idx, configs)
                                    for idx in range(n_reps)
                                    for configs in self.grid_config.gen_configs()]

        configs_chunks = self._chunks(configs_with_repetitions, n_models)

        self.jobs = (
            {'pipelines': self.pipelines,
             'functions': self.functions,
             'n_iters': n_iters,
             'configs': list(zip(*chunk))[1],
             'repetition': list(zip(*chunk))[0],
             'branches': branches,
             'name': name
            }
            for chunk in configs_chunks
        )

        self.n_jobs = ceil(len(configs_with_repetitions) / n_models)

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

    def run(self, n_reps, n_iters, n_workers=1, branches=1, name=None, progress_bar=False):
        """ Run research.

        Parameters
        ----------
        n_reps : int
            number of repetitions with each combination of parameters from grid_config.
        n_iters: int
            number of iterations for each configurations of each pipeline.
        n_workers : int, list of instances of Worker or list of dicts (Configs) (default 1).
            Workers (processes) to run tasks in parallel.
            If int - number of workers to run pipelines or workers that will run them, PipelineWorker will be used.
            If list of instances of Worker - workers to run tasks.
            If list of dicts (Configs) - list of additional configs which will be appended to configs from tasks.
                Each element corresponds to worker. Default Worker will be chosed as in case when n_workers is int.
        branches: int or list of dicts (Configs)
            Number of different configs which will use the same batch
            from root_pipeline. Pipelines will be executed in different threads.
            If int - number of pipelines with different configs that will use the same prepared batch
                from root_pipeline.
            If list of dicts (Configs) - list of dicts with additional configs to each pipeline.

            For example, if there are 2 GPUs, we can define parameter 'device' in model config as C('device')
            and define branches as [{'device': 0}, {'device': 1}].
        name : str or None
            name folder to save research. By default is 'research'.
        progress_bar : bool (default False)
            add tqdm progress bar

        At each iteration all add pipelines will be runned with some config from grid.
        """
        self.n_reps = n_reps
        self.n_iters = n_iters
        self.n_workers = n_workers
        self.branches = branches

        self.name = self._folder_exists(name)

        print("Research {} is starting...".format(self.name))

        self.save()

        self._create_jobs(n_reps, n_iters, branches, self.name)

        if isinstance(n_workers, int) or isinstance(n_workers[0], (dict, Config)):
            worker = PipelineWorker
        else:
            worker = None
        distr = Distributor(n_workers, worker)
        distr.run(self.jobs, dirname=self.name, n_jobs=self.n_jobs, progress_bar=progress_bar)
        return self

    def _folder_exists(self, name):
        name = name or 'research'
        if not os.path.exists(name):
            dirname = name
        else:
            i = 1
            while os.path.exists(name + '_' + str(i)):
                i += 1
            dirname = name + '_' + str(i)
            warnings.warn(
                "Research with name {} already exists. That research will be renamed to {}".format(name, dirname)
            )
        os.makedirs(dirname)
        return dirname

    def save(self):
        """ Save description of the research to folder name/description. """
        dirname = os.path.join(self.name, 'description')
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        with open(os.path.join(self.name, 'description', 'research.dill'), 'wb') as file:
            dill.dump(self, file)
        with open(os.path.join(self.name, 'description', 'research.json'), 'w') as file:
            file.write(json.dumps(self._json(), default=self._set_default_json))
        with open(os.path.join(self.name, 'description', 'alias.json'), 'w') as file:
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
        with open(os.path.join(name, 'description', 'research.dill'), 'rb') as file:
            return dill.load(file)
