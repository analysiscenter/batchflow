""" Class Research and auxiliary classes for multiple experiments. """

import os
import glob
from copy import copy, deepcopy
from collections import OrderedDict
from math import ceil
import json
import warnings
import dill
import pandas as pd

from .. import Config, Pipeline
from .distributor import Distributor
from .workers import PipelineWorker
from .grid import Grid
from .job import Job

class Research:
    """ Class Research for multiple parallel experiments with pipelines. """
    def __init__(self):
        self.executables = OrderedDict()
        self.loaded = False
        self.branches = 1
        self.trails = 3
        self.workers = 1
        self.progress_bar = False
        self.initial_name = 'research'
        self.n_reps = 1
        self.name = 'research'
        self.worker_class = PipelineWorker
        self.gpu = None
        self.n_jobs = None
        self.jobs = None
        self.grid_config = None
        self.n_iters = None
        self.timeout = 5

    def pipeline(self, root, branch=None, variables=None, name=None,
                 execute='%1', dump=-1, run=False, logging=False, **kwargs):
        """ Add new pipeline to research. Pipeline can be divided into root and branch. In that case root pipeline
        will prepare batch that can be used by different branches with different configs.

        Parameters
        ----------
        root : Pipeline
            a pipeline to execute when the research is run. It must contain `run` action with `lazy=True`.
            Only if `branch` is None, `root` may contain parameters that can be defined by grid.
        branch : Pipeline or None
            a parallelized pipeline to execute within the research.
            Several copies of branch pipeline will be executed in parallel per each batch
            received from the root pipeline.
            May contain parameters that can be defined by grid.
        variables : str, list of str or None
            names of pipeline variables to save after each iteration into results. All of them must be
            defined in `root`

            If `branch` is None or be defined in `branch` if `branch` is not None.
            If None, pipeline will be executed without any dumping
        name : str
            pipeline name. If None, pipeline will have name `pipeline_{index}`
        execute : int, str or list of int or str
            If -1, pipeline will be executed just at last iteration (if `iteration + 1 == n_iters`
            or `StopIteration` was raised)

            If positive int, pipeline will be excuted for that iteration

            If str, must be `'%{step}'` where step is int

            If list, must be list of int or str descibed above
        dump : int, str or list of int or str
            iteration when results will be dumped and cleared. Similar to execute
        run : bool
            if False then `.next_batch()` will be applied to pipeline, else `.run()` and then `.reset_iter()`.
        kwargs :
            parameters in pipeline config that depends on the names of the other pipeline.

            For example,
            if test pipeline imports model from the other pipeline with name `'train'` in Research,
            corresponding parameter in import_model must be `C('import_from')` and add_pipeline
            must be called with parameter `import_from='train'`.
        logging : bool
            include execution information to log file or not


        **How to define changing parameters**

        All parameters in `root` or `branch` that are defined in grid should be defined
        as `C('parameter_name')`. Corresponding parameter in grid must have the same `'parameter_name'`.
        """
        name = name or 'pipeline_' + str(len(self.executables) + 1)

        if name in self.executables:
            raise ValueError('Executable unit with name {} was alredy existed'.format(name))

        unit = Executable()
        unit.add_pipeline(root, name, branch, variables,
                          execute, dump, run, logging, **kwargs)
        self.executables[name] = unit
        return self

    def function(self, function, returns=None, name=None, execute='%1', dump=-1,
                 on_root=False, logging=False, *args, **kwargs):
        """ Add function to research.

        Parameters
        ----------
        function : callable
            callable object with following parameters:
                experiment : `OrderedDict` of Executable
                    all pipelines and functions that were added to Research
                iteration : int
                    iteration when function is called
                **args, **kwargs
        returns : str, list of str or None
            names for function returns to save into results
            if None, `function` will be executed without any saving results and dumping
        name : str (default None)
            function name. If None, a function will have name `func_{index}`
        execute : int, str or list of int or str
            If -1, function will be executed just at last iteration (if `iteration + 1 == n_iters`
            or `StopIteration` was raised)

            If positive int, function will be excuted for that iteration

            If str, must be `'%{step}'` where step is int

            If list, must be list of int or str descibed above
        dump : int, str or list of int or str
            iteration when results will be dumped and cleared. Similar to execute
        on_root : bool
            if False, function will be called with parameters `(iteration, experiment, *args, **kwargs)`,
            else with `(iteration, experiments, *args, **kwargs)` where `experiments` is a list of single experiments
        logging : bool
            include execution information to log file or not
        args : list
            args for the function
        kwargs : dict
            kwargs for the function


        **How to use experiment**

        Each pipeline and function added to Research is saved as an :class:`~.Executable`.

        Experiment is an `OrderedDict` of all pipelines and functions that were added to Research
        and are running in current Job. Key is a name of `Executable`, value is `Executable`.
        """

        name = name or 'func_' + str(len(self.executables) + 1)

        if name in self.executables:
            raise ValueError('Executable unit with name {} was alredy existed'.format(name))

        if on_root and returns is not None:
            raise ValueError("If function on root, then it mustn't have returns")

        unit = Executable()
        unit.add_function(function, name, execute, dump,
                          returns, on_root, logging, *args, **kwargs)
        self.executables[name] = unit

        return self

    def grid(self, grid_config):
        """ Add grid of pipeline parameters. Configs from that grid will be generated
        and then substitute into pipelines.

        Parameters
        ----------
        grid_config : dict, Grid or Option
            if dict it should have items parameter_name: list of values.
        """
        self.grid_config = Grid(grid_config)
        return self

    def load_results(self, *args, **kwargs):
        """ Load results of research as pandas.DataFrame (see Results.load). """
        return Results(research=self).load(*args, **kwargs)

    def _create_jobs(self, n_reps, n_iters, branches, name):
        """ Create generator of jobs. If `branches=1` or `len(branches)=1` then each job is one repetition
        for each config from grid_config. Else each job contains several pairs `(repetition, config)`.

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

        jobs = (Job(self.executables, n_iters,
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

    def run(self, n_reps=1, n_iters=None, workers=1, branches=1, name=None,
            progress_bar=False, gpu=None, worker_class=None, timeout=5, trails=2):
        """ Run research.

        Parameters
        ----------
        n_reps : int
            number of repetitions with each combination of parameters from `grid_config`
        n_iters: int or None
            number of iterations for each configurations. If None, wait StopIteration exception.
        workers : int or list of dicts (Configs)
            If int - number of workers to run pipelines or workers that will run them, `PipelineWorker` will be used.

            If list of dicts (Configs) - list of additional configs which will be appended to configs from tasks.

            Each element corresponds to one worker.
        branches: int or list of dicts (Configs)
            Number of different branches with different configs with the same root. Each branch will use the same batch
            from `root`. Pipelines will be executed in different threads.

            If int - number of pipelines with different configs that will use the same prepared batch
            from `root`.

            If list of dicts (Configs) - list of dicts with additional configs to each pipeline.
        name : str or None
            name folder to save research. By default is 'research'.
        progress_bar : bool
            add tqdm progress bar
        gpu : str, list or None
            all gpu devices available for the research.

            Must be of length 1 or be divisible
            by the number of workers.

            If is divisible by the number of workers then
            `length / n_workers` must be 1 or be divisible by the number of branches. If you want to use different
            devices in branches, use expression `C('device')`.

            For example, for :class:`~.TFModel` add `device=C('device')` to model config.
            if None, default gpu configuration will be used
        timeout : int
            each job will be killed if it doesn't answer more then that time in minutes
        trails : int
            trails to execute job


        **How does it work**

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
            gpu = []
        elif isinstance(gpu, str):
            gpu = [int(item) for item in gpu.split(',')]
        else:
            gpu = gpu
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
            x = json.dumps(obj)
        except TypeError:
            x = str(obj)
        return x

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

class Executable:
    """ Function or pipeline

    **Attributes**

    function : callable
        is None if `Executable` is a pipeline
    pipeline : Pipeline
        is None if `Executable` is a function
    root_pipeline : Pipeline
        is None if `Executable` is a function or pipeline is not divided into root and branch
    result : dict
        current results of the `Executable`. Keys are names of variables (for pipeline)
        or returns (for function) values are lists of variable values
    path : str
        path to the folder where results will be dumped
    exec : int, list of ints or None
    dump : int, list of ints or None
    to_run : bool
    variables : list
        variables (for pipeline) or returns (for function)
    on_root : bool

    args : list

    kwargs : dict()
    """
    def __init__(self):
        self.function = None
        self.pipeline = None
        self.name = None
        self.result = None
        self.execute = None
        self.dump = None
        self.to_run = None
        self.variables = []
        self.root_pipeline = None
        self.on_root = None
        self.args = []
        self.kwargs = dict()
        self.path = None
        self.config = None
        self.logging = None
        self.additional_config = None
        self.action = None

    def add_function(self, function, name, execute='%1', dump=-1, returns=None,
                     on_root=False, logging=False, *args, **kwargs):
        """ Add function as an Executable Unit. """
        returns = returns or []

        if not isinstance(returns, list):
            returns = [returns]

        self.name = name
        self.function = function
        self.execute = execute
        self.dump = dump
        self.variables = returns
        self.args = args
        self.kwargs = kwargs
        self.on_root = on_root
        self.logging = logging

        self.action = {
            'type': 'function',
            'name': name,
            'on_root': on_root
        }

        self._clear_result()
        self._process_iterations()

    def add_pipeline(self, root_pipeline, name, branch_pipeline=None, variables=None,
                     execute='%1', dump=-1, run=False, logging=False, **kwargs):
        """ Add pipeline as an Executable Unit """
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
        self.execute = execute
        self.dump = dump
        self.to_run = run
        self.kwargs = kwargs
        self.logging = logging

        self.action = {
            'type': 'pipeline',
            'name': name,
            'root': root is not None,
            'run': run
        }

        self.config = None

        self.additional_config = None

        self._clear_result()
        self._process_iterations()

    def _process_iterations(self):
        if not isinstance(self.execute, list):
            self.execute = [self.execute]
        if not isinstance(self.dump, list):
            self.dump = [self.dump]

    def get_copy(self):
        """ Create copy of unit """
        new_unit = copy(self)
        if self.pipeline is not None:
            new_unit.pipeline += Pipeline()
        new_unit.result = deepcopy(new_unit.result)
        new_unit.variables = copy(new_unit.variables)
        return new_unit

    def reset_iter(self):
        """ Reset iterators in pipelines """
        if self.pipeline is not None:
            self.pipeline.reset_iter()
        if self.root_pipeline is not None:
            self.root_pipeline.reset_iter()

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
            batch = self.pipeline.next_batch()
        else:
            raise TypeError("Executable should be pipeline, not a function")
        return batch

    def run(self):
        """ Run pipeline """
        if self.pipeline is not None:
            self.pipeline.reset_iter()
            self.pipeline.run()
        else:
            raise TypeError("Executable should be pipeline, not a function")

    def reset_root_iter(self):
        """ Reset pipeline iterator """
        if self.root_pipeline is not None:
            self.root_pipeline.reset_iter()
        else:
            raise TypeError("Executable must have root")

    def next_batch_root(self):
        """ Next batch from root pipeline """
        if self.root_pipeline is not None:
            batch = self.root_pipeline.next_batch()
        else:
            raise TypeError("Executable should have root pipeline")
        return batch

    def execute_for(self, batch, iteration):
        """ Execute pipeline for batch from root """
        _ = iteration
        if self.pipeline is not None:
            batch = self.pipeline.execute_for(batch)
        else:
            raise TypeError("Executable should be pipeline, not a function")
        return batch

    def _call_pipeline(self, iteration, *args, **kwargs):
        _ = args, kwargs
        if self.to_run:
            self.run()
        else:
            self.next_batch()
        self.put_result(iteration)

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
        """ Put result from pipeline to self.result """
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
        rule = self.execute if action == 'execute' else self.dump
        list_rule = [item for item in rule if isinstance(item, int)]
        step_rule = [int(item[1:]) for item in rule if isinstance(item, str)]

        in_list = iteration in list_rule
        in_step = sum([(iteration+1) % item == 0 for item in step_rule])

        if n_iters is None:
            action_list = in_list or in_step
        else:
            in_final = -1 in list_rule and iteration+1 == n_iters
            action_list = in_list or in_step or in_final
        return action_list

class Results():
    """ Class for dealing with results of research

    Parameters
    ----------
    path : str
        path to root folder of research
    research : Research
        instance of Research
    """
    def __init__(self, path=None, research=None):
        if path is None and research is None:
            raise ValueError('At least one of parameters path and research must be not None')
        if path is None:
            self.research = research
            self.path = research.name
        else:
            self.research = Research().load(path)
            self.path = path

    def _get_list(self, value):
        if not isinstance(value, list):
            value = [value]
        return value

    def _sort_files(self, files, iterations):
        files = {file: int(file.split('_')[-1]) for file in files}
        files = OrderedDict(sorted(files.items(), key=lambda x: x[1]))
        result = []
        start = 0
        for name, end in files.items():
            intersection = pd.np.intersect1d(iterations, pd.np.arange(start, end))
            if len(intersection) > 0:
                result.append((name, intersection))
            start = end
        return OrderedDict(result)

    def _slice_file(self, dumped_file, iterations_to_load, variables):
        iterations = dumped_file['iteration']
        if len(iterations) > 0:
            elements_to_load = pd.np.array([pd.np.isin(it, iterations_to_load) for it in iterations])
            res = OrderedDict()
            for variable in ['iteration', *variables]:
                if variable in dumped_file:
                    res[variable] = pd.np.array(dumped_file[variable])[elements_to_load]
        else:
            res = None
        return res

    def _concat(self, results, variables):
        res = {key: [] for key in [*variables, 'iteration']}
        for chunk in results:
            if chunk is not None:
                for key, values in res.items():
                    if key in chunk:
                        values.extend(chunk[key])
        return res

    def _fix_length(self, chunk):
        max_len = max([len(value) for value in chunk.values()])
        for value in chunk.values():
            if len(value) < max_len:
                value.extend([pd.np.nan] * (max_len - len(value)))
        return max_len

    def _filter_configs(self, config=None, alias=None):
        result = None
        if config is None and alias is None:
            raise ValueError('At least one of parameters config and alias must be not None')
        if config is not None:
            result = self.configs.subset(config, by_alias=False)
        else:
            result = self.configs.subset(alias, by_alias=True)
        return result


    def load(self, names=None, repetitions=None, variables=None, iterations=None,
             configs=None, aliases=None, use_alias=False, as_dataframe=True):
        """ Load results as pandas.DataFrame.

        Parameters
        ----------
        names : str, list or None
            names of units (pipleines and functions) to load
        repetitions : int, list or None
            numbers of repetitions to load
        variables : str, list or None
            names of variables to load
        iterations : int, list or None
            iterations to load
        configs, aliases : dict, Config, Option, Grid or None
            configs to load
        use_alias : bool
            if True, the resulting DataFrame/dict will have one column/item with alias, else it will
            have column/item for each option in grid
        as_dataframe : bool
            return pandas.DataFrame or dict

        Return
        ------
        pandas.DataFrame or dict
            will have columns/keys: iteration, repetition, name (of pipeline/function)
            and column/key for config. Also it will have column/key for each variable of pipeline
            and output of the function that was saved as a result of the research.
        """
        self.configs = self.research.grid_config
        transform = lambda x: pd.DataFrame(x) if as_dataframe else x
        if configs is None and aliases is None:
            self.configs = list(self.configs.gen_configs())
        elif configs is not None:
            self.configs = self._filter_configs(config=configs)
        else:
            self.configs = self._filter_configs(alias=aliases)

        if names is None:
            names = list(self.research.executables.keys())

        if repetitions is None:
            repetitions = list(range(self.research.n_reps))

        if variables is None:
            variables = [variable for unit in self.research.executables.values() for variable in unit.variables]

        if iterations is None:
            iterations = list(range(self.research.n_iters))

        self.names = self._get_list(names)
        self.repetitions = self._get_list(repetitions)
        self.variables = self._get_list(variables)
        self.iterations = self._get_list(iterations)

        all_results = []

        for config_alias in self.configs:
            alias = config_alias.alias(as_string=False)
            alias_str = config_alias.alias(as_string=True)
            for repetition in self.repetitions:
                for unit in self.names:
                    path = os.path.join(self.path, 'results', alias_str, str(repetition))
                    files = glob.glob(os.path.join(glob.escape(path), unit + '_[0-9]*'))
                    files = self._sort_files(files, self.iterations)
                    if len(files) != 0:
                        res = []
                        for filename, iterations_to_load in files.items():
                            with open(filename, 'rb') as file:
                                res.append(self._slice_file(dill.load(file), iterations_to_load, self.variables))
                        res = self._concat(res, self.variables)
                        max_len = self._fix_length(res)
                        if use_alias:
                            all_results.append(
                                {
                                    'config': [alias_str] * max_len,
                                    'repetition': [repetition] * max_len,
                                    'name': [unit] * max_len,
                                    **res
                                }
                                )
                        else:
                            _alias = {key: [value] * max_len for key, value in alias.items()}
                            all_results.append(
                                {
                                    **_alias,
                                    'repetition': [repetition] * max_len,
                                    'name': [unit] * max_len,
                                    **res
                                }
                                )
        if len(all_results) > 0:
            concat_results = {key: [] for key in all_results[0]}
            for key in concat_results:
                for result in all_results:
                    concat_results[key] += result[key]
        else:
            concat_results = None
        return transform(concat_results)
