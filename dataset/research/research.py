#pylint:disable=too-many-instance-attributes
#pylint:disable=too-few-public-methods
#pylint:disable=attribute-defined-outside-init
#pylint:disable=bare-except
#pylint:disable=no-value-for-parameter

""" Class Research and auxiliary classes for multiple experiments. """

import os
from copy import copy, deepcopy
from collections import OrderedDict
from math import ceil
import json
import warnings
import dill

from .. import Config, Pipeline
from .distributor import Distributor
from .workers import PipelineWorker
from .grid import Grid
from .job import Job

class Research:
    """ Class Research for multiple parallel experiments with pipelines. """
    def __init__(self):
        self.executable_units = OrderedDict()

    def pipeline(self, root_pipeline, branch_pipeline=None, variables=None, name=None,
                 execute_for=1, dump_for=-1, run=False, **kwargs):
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
            If None, pipeline results will be dumped at last iteration.
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

        if name in self.executable_units:
            raise ValueError('Executable unit with name {} was alredy existed'.format(name))

        unit = ExecutableUnit()
        unit.add_pipeline(root_pipeline, name, branch_pipeline, variables,
                          execute_for, dump_for, run, **kwargs)
        self.executable_units[name] = unit
        return self

    def function(self, function, name, execute_for=1, dump_for=-1, returns=None, on_root=False, *args, **kwargs):
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

        if name in self.executable_units:
            raise ValueError('Executable unit with name {} was alredy existed'.format(name))

        unit = ExecutableUnit()
        unit.add_function(function, name, execute_for, dump_for,
                          returns, on_root, *args, **kwargs)
        self.executable_units[name] = unit

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

        jobs = (Job(self.executable_units, n_iters, 
                    list(zip(*chunk))[0], list(zip(*chunk))[1], branches, name)
            for chunk in configs_chunks
        )

        n_jobs = ceil(len(configs_with_repetitions) / n_models)

        return jobs, n_jobs

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

    def run(self, n_reps, n_iters, workers=1, branches=1, name=None, progress_bar=False, gpu=None, worker_class=None, timeout=5):
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
        self.workers = workers
        self.branches = branches
        self.progress_bar = progress_bar
        self.gpu = self._get_gpu_list(gpu)
        self.worker_class = worker_class or PipelineWorker
        self.timeout = timeout

        n_workers = workers if isinstance(workers, int) else len(workers)
        n_branches = branches if isinstance(branches, int) else len(branches)

        if len(self.gpu) > 1 and len(self.gpu) % n_workers != 0:
            raise ValueError("Number of gpus must be 1 or be divisible \
                             by the number of workers but {} was given".format(len(self.gpu)))

        if len(self.gpu) > 1 and len(self.gpu) // n_workers > 1 and (len(self.gpu) // n_workers) % n_branches != 0:
            raise ValueError("Number of gpus / n_workers must be 1 \
                             or be divisible by the number of branches but {} was given".format(len(self.gpu)))

        self.name = self._folder_exists(name)

        print("Research {} is starting...".format(self.name))

        self.save()

        self.jobs, self.n_jobs = self._create_jobs(n_reps, n_iters, branches, self.name)

        distr = Distributor(workers, self.gpu, self.worker_class, self.timeout)
        distr.run(self.jobs, dirname=self.name, n_jobs=self.n_jobs, progress_bar=progress_bar)
        return self

    def _get_gpu_list(self, gpu):
        if gpu is None:
            return []
        elif isinstance(gpu, str):
            return [int(item) for item in gpu.split(',')]
        else:
            return gpu

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
        # for name, pipeline in self.pipelines.items():
        #     _pipelines[name] = copy(pipeline)
        #     del _pipelines[name]['ppl']
        # description['pipelines'] = _pipelines
        return description

    @classmethod
    def load(cls, name):
        """ Load description of the research from name/description. """
        with open(os.path.join(name, 'description', 'research.dill'), 'rb') as file:
            return dill.load(file)

class ExecutableUnit:
    """ Function or pipeline. """
    def __init__(self):
        self.function = None
        self.pipeline = None
        self.result = None
        self.to_run = None
        self.root_pipeline = None
        self.on_root = None
        self.args = []
        self.kwargs = dict()
        self.path = None

    def add_function(self, function, name, execute_for=1, dump_for=-1, returns=None, on_root=False, *args, **kwargs):
        """ Add function as a Executable Unit. """
        returns = returns or []        

        if not isinstance(returns, list):
            returns = [returns]

        self.name = name
        self.function = function
        self.exec_for = execute_for
        self.dump_for = dump_for
        self.variables = returns
        self.args = args
        self.kwargs = kwargs
        self.on_root = on_root

        self._init_result()

    def add_pipeline(self, root_pipeline, name, branch_pipeline=None, variables=None,
                 execute_for=1, dump_for=-1, run=False, **kwargs):
        """ Add pipeline as a Executable Unit """
        variables = variables or []

        if not isinstance(variables, list):
            variables = [variables]

        if branch_pipeline is None:
            pipeline = root_pipeline
            root = None
        else:
            pipeline = branch_pipeline
            root = root_pipeline

        self.name = name
        self.pipeline = pipeline
        self.root_pipeline = root
        self.variables = variables
        self.exec_for = execute_for
        self.dump_for = dump_for
        self.to_run = run
        self.kwargs = kwargs

        self.config = None

        self.additional_config = None

        self._init_result()

    def get_copy(self):
        new_unit = copy(self)
        if self.pipeline is not None:
            # new_unit.name = self.name
            # new_unit.pipeline = self.pipeline
            # new_unit.root_pipeline = self.root
            # new_unit.variables = self.variables
            # new_unit.execute_for = self.execute_for
            # new_unit.dump_for = self.dump_for
            # new_unit.run = self.run
            # new_unit.kwargs = self.kwargs
            # new_unit.config = self.config
            # new_unit.additional_config = self.additional_config
            new_unit.pipeline += Pipeline()
        new_unit.result = deepcopy(new_unit.result)
        new_unit.variables = copy(new_unit.variables)
        return new_unit

    def _init_result(self):
        self.result = {var: [] for var in self.variables}
        self.result['iteration'] = []

    def set_config(self, config, worker_config, branch_config, import_config):
        """ Set new config for pipeline """
        self.config = config
        self.additional_config = Config(worker_config) + Config(branch_config) + Config(import_config)

        if self.pipeline is not None:
            self.pipeline.set_config(config.config() + self.additional_config)

    def next_batch(self):
        if self.pipeline is not None:
            return self.pipeline.next_batch()
        else:
            raise TypeError("ExecutableUnit should be pipeline, not a function")

    def run(self):
        if self.pipeline is not None:
            self.pipeline.reset_iter()
            self.pipeline.run()
        else:
            raise TypeError("ExecutableUnit should be pipeline, not a function")

    def next_batch_root(self):
        if self.root_pipeline is not None:
            return self.root_pipeline.next_batch()
        else:
            raise TypeError("ExecutableUnit should have root pipeline")

    def execute_for(self, batch, iteration):
        if self.pipeline is not None:
            batch = self.pipeline.execute_for(batch)
            # self.put_result(iteration)
            return batch
        else:
            raise TypeError("ExecutableUnit should be pipeline, not a function")

    def _call_pipeline(self, iteration, *args, **kwargs):
        if self.to_run:
            result = self.run()
        else:
            result = self.next_batch()
        self.put_result(iteration)
        return result

    def _call_function(self, iteration, *args, **kwargs):
        result = self.function(iteration, *args, **kwargs)
        self.put_result(iteration, result)
        return result

    def __call__(self, iteration, *args, **kwargs):
        if self.pipeline is not None:
            self._call_pipeline(iteration, *args, **kwargs)
        else:
            self._call_function(iteration, *args, **kwargs)

    def put_result(self, iteration, result=None):
        if len(self.variables) > 0:
            if self.pipeline is not None:
                for variable in self.variables:
                    self.result[variable].append(self.pipeline.get_variable(variable))
            else:
                if len(self.variables) == 1:
                    result = [result]
                for variable, value in zip(self.variables, result):
                    self.result[variable].append(value)
            self.result['iteration'].append(iteration)

    def dump_result(self, filename):
        """ Dump pipeline results. """
        if len(self.variables) > 0:
            path = os.path.join(self.path, filename)
            with open(path, 'wb') as file:
                dill.dump(self.result, file)

    def create_folder(self, name):
        self.path = os.path.join(name, 'results', self.config.alias(as_string=True), str(self.repetition))
        if not os.path.exists(self.path):
            os.makedirs(self.path)

