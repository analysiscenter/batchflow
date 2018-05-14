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
        self.loaded = False

    def pipeline(self, root_pipeline, branch_pipeline=None, variables=None, name=None,
                 execute_for=1, dump_for=-1, run=False, **kwargs):
        """ Add new pipeline to research. Pipeline can be divided into root and branch. In that case root pipeline
        will prepare batch that can be used by different branches with different configs.

        Parameters
        ----------
        root_pipeline : dataset.Pipeline
            root_pipeline must have run action with lazy=True. If branch_pipeline is None then root_pipeline
            may contain parameters that can be defined by grid.
        branch_pipeline : dataset.Pipeline or None
            if not None, for resulting batch from root_pipeline branch_pipeline.execute_for(batch) will be called.
            May contain parameters that can be defined by grid.
        variables : str, list of str or None
            names of pipeline variables to save after each iteration into results. All of them must be
            defined in root_pipeline
            if branch_pipeline is None or be defined  in branch_pipeline if branch_pipeline is not None.
            if None, pipeline will be executed without any dumping
        name : str (default None)
            pipeline name inside research. If name is None, pipeline will have name 'ppl_{index}'
        execute_for : int, list of ints
            If -1, pipeline will be executed just at last iteration.
            If positive int, pipeline will be excuted for iterations with that step
            If list of ints, pipeline will be excuted for that iterations
            If None, pipeline will executed at each iteration.
        dump_for : int, list of ints
            iteration when results will be dumped and cleared. Similar to execute_for
            If None, pipeline results will be dumped at last iteration.
        run : bool (default False)
            if False then .next_batch() will be applied to pipeline, else .run() and then reset_iter().
        kwargs :
            parameters in pipeline config that depends on the names of the other pipeline. For example,
            if test pipeline imports model from the other pipeline with name 'train' in Research,
            corresponding parameter in import_model must be C('import_from') and add_pipeline
            must be called with parameter import_from='train'.

        **How to define changing parameters**

        All parameters in root_pipeline or branch_pipeline that are defined in grid should be defined
        as C('parameter_name'). Corresponding parameter in grid must have the same 'parameter_name'.
        """
        name = name or 'unit_' + str(len(self.executable_units))

        if name in self.executable_units:
            raise ValueError('Executable unit with name {} was alredy existed'.format(name))

        unit = ExecutableUnit()
        unit.add_pipeline(root_pipeline, name, branch_pipeline, variables,
                          execute_for, dump_for, run, **kwargs)
        self.executable_units[name] = unit
        return self

    def function(self, function, returns=None, name=None, execute_for=1, dump_for=-1, on_root=False, *args, **kwargs):
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
        returns : str, list of str or None
            names for function returns to save into results
            if None, function will be executed without any dumping
        name : str (default None)
            function name inside research. If name is None, pipeline will have name 'func_{index}'
        execute_for : int, list of ints
            If -1, function will be called just at last iteration.
            If positive int, function will be called for iterations with that step
            If list of ints, function will be called for that iterations
            If None, function will called at each iteration
        dump_for : int, list of ints
            iteration when results will be dumped. Similar to execute_for
            If None, function results will not be dumped
        on_root : bool
            if False, function will be called with parameters (iteration, experiment, *args, **kwargs),
            else with  (iteration, experiments, *args, **kwargs) where experiments is a list of instances
            of Experiment corresponding to all branches
        args, kwargs :
            args and kwargs for the function

        **How to use experiment**
        Experiment is an OrderedDict for all pipelines and functions that were added to Research
        and are running in current Job. Key is a name of ExecutableUnit, value is ExecutableUnit.

        Each pipeline and function added to Research is saved as an ExecutableUnit. Each ExecutableUnit
        has the following attributes:

            function : callable
                is None if ExecutableUnit is a pipeline
            pipeline : Pipeline
                is None if ExecutableUnit is a function
            root_pipeline : Pipeline
                is None if ExecutableUnit is a function or pipeline is not divided into root and branch
            result : dict
                current results of the ExecutableUnit. Keys are names of variables (for pipeline)
                or returns (for function) values are lists of variable values
            path : str
                path to the folder where results will be dumped
            exec_for : int, list of ints or None
            dump_for : int, list of ints or None
            to_run : bool
            variables : list
                variables (for pipeline) or returns (for function)
            on_root : bool
            args : list
            kwargs : dict()
        """

        name = name or 'unit_' + str(len(self.executable_units))

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
            chunk size
        """
        for i in range(0, len(array), size):
            yield array[i:i + size]

    def run(self, n_reps=1, n_iters=100, workers=1, branches=1, name=None,
            progress_bar=False, gpu=None, worker_class=None, timeout=5, trails=2):
        """ Run research.

        Parameters
        ----------
        n_reps : int
            number of repetitions with each combination of parameters from grid_config
        n_iters: int
            number of iterations for each configurations
        workers : int or list of dicts (Configs) (default 1)
            Workers (processes) to run tasks in parallel.
            If int - number of workers to run pipelines or workers that will run them, PipelineWorker will be used.
            If list of dicts (Configs) - list of additional configs which will be appended to configs from tasks.
            Each element corresponds to one worker.
        branches: int or list of dicts (Configs)
            Number of different branches with different configs with the same root. Each branch will use the same batch
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
        gpu : str, list or None
            all gpu devices available for the research. Must be of length 1 or be divisible
            by the number of workers. If is divisible by the number of workers then
            length / n_workers must be 1 or be divisible by the number of branches. If you want to use different
            devices in branches, use expression C('device'). For example, for TFModel add device=C('device')
            to model config.
            if None, default gpu configuration will be used
        timeout : int
            time in minutes
            each job will be killed if it doesn't answer more then that time
        trails : int
            trials to execute job

        At each iteration all add pipelines will be runned with some config from grid.

        ** How it works **
        At each iteration all pipelines and functions will be executed in the order in which were added.
        """
        if not self.loaded:
            self.n_reps = n_reps
            self.n_iters = n_iters
            self.workers = workers
            self.branches = branches
            self.progress_bar = progress_bar
            self.gpu = self._get_gpu_list(gpu)
            self.worker_class = worker_class or PipelineWorker
            self.timeout = timeout
            self.trails = trails
            self.initial_name = name
            self.name = name

        n_workers = self.workers if isinstance(self.workers, int) else len(self.workers)
        n_branches = self.branches if isinstance(self.branches, int) else len(self.branches)

        if len(self.gpu) > 1 and len(self.gpu) % n_workers != 0:
            raise ValueError("Number of gpus must be 1 or be divisible \
                             by the number of workers but {} was given".format(len(self.gpu)))

        if len(self.gpu) > 1 and len(self.gpu) // n_workers > 1 and (len(self.gpu) // n_workers) % n_branches != 0:
            raise ValueError("Number of gpus / n_workers must be 1 \
                             or be divisible by the number of branches but {} was given".format(len(self.gpu)))

        self.name = self._folder_exists(self.initial_name)

        print("Research {} is starting...".format(self.name))

        self.save()

        self.jobs, self.n_jobs = self._create_jobs(self.n_reps, self.n_iters, self.branches, self.name)

        distr = Distributor(self.workers, self.gpu, self.worker_class, self.timeout, self.trails)
        distr.run(self.jobs, dirname=self.name, n_jobs=self.n_jobs,
                  n_iters=self.n_iters, progress_bar=self.progress_bar)
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
        return description

    @classmethod
    def load(cls, name):
        """ Load description of the research from name/description. """
        with open(os.path.join(name, 'description', 'research.dill'), 'rb') as file:
            res = dill.load(file)
            res.loaded = True
            return res

class ExecutableUnit:
    """ Function or pipeline. """
    def __init__(self):
        self.function = None
        self.pipeline = None
        self.result = None
        self.exec_for = None
        self.dump_for = None
        self.to_run = None
        self.variables = []
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

        self._clear_result()

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

        self._clear_result()

    def get_copy(self):
        """ Create copy of unit """
        new_unit = copy(self)
        if self.pipeline is not None:
            new_unit.pipeline += Pipeline()
        new_unit.result = deepcopy(new_unit.result)
        new_unit.variables = copy(new_unit.variables)
        return new_unit

    def _clear_result(self):
        self.result = {var: [] for var in self.variables}
        self.result['iteration'] = []

    def set_config(self, config, worker_config, branch_config, import_config):
        """ Set new config for pipeline """
        self.config = config
        self.additional_config = Config(worker_config) + Config(branch_config) + Config(import_config)

        if self.pipeline is not None:
            self.pipeline.set_config(config.config() + self.additional_config)

    def next_batch(self):
        """ Next batch from pipeline """
        if self.pipeline is not None:
            return self.pipeline.next_batch()
        else:
            raise TypeError("ExecutableUnit should be pipeline, not a function")

    def run(self):
        """ Run pipeline """
        if self.pipeline is not None:
            self.pipeline.reset_iter()
            self.pipeline.run()
        else:
            raise TypeError("ExecutableUnit should be pipeline, not a function")

    def next_batch_root(self):
        """ Next batch from root pipeline """
        if self.root_pipeline is not None:
            batch = self.root_pipeline.next_batch()
            return batch
        else:
            raise TypeError("ExecutableUnit should have root pipeline")

    def execute_for(self, batch, iteration):
        """ Execute pipeline for batch from root """
        _ = iteration
        if self.pipeline is not None:
            batch = self.pipeline.execute_for(batch)
            return batch
        else:
            raise TypeError("ExecutableUnit should be pipeline, not a function")

    def _call_pipeline(self, iteration, *args, **kwargs):
        _ = args, kwargs
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
        """ Put result from pipeline to self.results """
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

    def dump_result(self, iteration, filename):
        """ Dump pipeline results """
        if len(self.variables) > 0:
            path = os.path.join(self.path, filename + '_' + str(iteration))
            with open(path, 'wb') as file:
                dill.dump(self.result, file)
        self._clear_result()

    def create_folder(self, name):
        """ Create folder if it doesn't exist """
        self.path = os.path.join(name, 'results', self.config.alias(as_string=True), str(self.repetition))
        if not os.path.exists(self.path):
            os.makedirs(self.path)

    def action_iteration(self, iteration, n_iters=None, action='execute'):
        """ Returns does Unit should be executed at that iteration """
        rule = self.exec_for if action == 'execute' else self.dump_for
        list_rule = isinstance(rule, list) and iteration in rule
        step_rule = isinstance(rule, int) and rule > 0 and (iteration+1) % rule == 0
        if n_iters is None:
            return list_rule or step_rule
        else:
            final_rule = isinstance(rule, int) and rule == -1 and iteration+1 == n_iters
            return list_rule or step_rule or final_rule
