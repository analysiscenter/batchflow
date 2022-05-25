""" Storages of research configs, states and results. """

import logging
import os
import sys
import re
import glob
import json
import io
import subprocess
import contextlib
from collections import OrderedDict
import dill
import multiprocess as mp

from .profiler import ExperimentProfiler, ResearchProfiler
from .results import ResearchResults
from .utils import to_list, create_logger, jsonify, create_output_stream

class BaseExperimentStorage:
    """ Storage for experiment data.

    Parameters
    ----------
    experiment : Experiment

    loglevel : str, optional
        logging level, by default 'error'
    storage : str, optional
        type of the storage, by default 'memory'
    """
    def __new__(cls, *args, storage='memory', **kwargs):
        _ = args, kwargs
        if storage == 'local':
            return super().__new__(LocalExperimentStorage)
        if storage == 'memory':
            return super().__new__(MemoryExperimentStorage)
        if storage == 'clearml':
            return super().__new__(ClearMLExperimentStorage)

        raise ValueError(f'Unknown storage mode: {storage}')

    def __init__(self, experiment, loglevel=None, storage='memory'):
        _ = storage
        self.experiment = experiment
        self.loglevel = loglevel or 'error'

        self.logger = None
        self.results = OrderedDict()
        self.stdout_file = None
        self.stderr_file = None
        self.profiler = None

        self._create_profiler()

    def update_variable(self, name, value):
        """ Set value of the variable for the current iteration. """
        results = self.results.get(name, OrderedDict())
        results[self.experiment.iteration] = value
        self.results[name] = results

    def close(self):
        raise AttributeError('`close` method must be defined.')

    def _create_profiler(self):
        profile = self.experiment.profile
        if profile == 2 or isinstance(profile, str) and 'detailed'.startswith(profile):
            self.profiler = ExperimentProfiler(detailed=True)
        elif profile == 1 or profile is True:
            self.profiler = ExperimentProfiler(detailed=False)
        else: # 0, False, None
            self.profiler = None

    @property
    def name(self):
        """ Name of the research (if exists) or experiment. """
        if self.experiment.research:
            name = self.experiment.research.name
        else:
            name = self.experiment.executor.name
        return name

class MemoryExperimentStorage(BaseExperimentStorage):
    """ Experiment storage in RAM without any dumping on disk.

    Parameters
    ----------
    experiment : Experiment

    loglevel : str, optional
        logging level, by default 'error'
    """
    def __init__(self, experiment, loglevel=None, storage='memory'):
        super().__init__(experiment, loglevel, storage)

        self.create_redirection_streams()
        self._create_logger()

    def close(self):
        """ Close streams and loggers and send results to the common research storage. """
        self._update_research_results()
        self._update_research_profiler()
        self._close_files()
        self._close_logger()

    # Initialization methods

    def create_redirection_streams(self):
        """ Create streams to redirect stdout and stderr (if needed). """
        self.stdout_file = create_output_stream(self.experiment.redirect_stdout, False, common=False)
        self.stderr_file = create_output_stream(self.experiment.redirect_stderr, False, common=False)

    def _create_logger(self):
        """ Create logger. """
        logger_name = self.name + '.' + self.experiment.id
        self.logger = create_logger(logger_name, None, self.loglevel)

    # Closing methods

    def _update_research_results(self):
        """ Copy experiment results to the common research storage. """
        experiment = self.experiment
        if experiment.research is not None:
            experiment.research.results.put(experiment.id, experiment.results, experiment.config_alias)

    def _update_research_profiler(self):
        """ Copy profiler stats to the common research storage. """
        experiment = self.experiment
        if experiment.research is not None and self.profiler is not None:
            experiment.research.profiler.put(experiment.id, self.profiler.profile_info)

    def _close_files(self):
        """ Close stdout/stderr files (if rederection was performed). """
        for name in ['stdout', 'stderr']:
            file = getattr(self, name+'_file')
            if not isinstance(file, (contextlib.nullcontext, type(None))):
                if isinstance(file, io.StringIO) and not file.closed:
                    content = file.getvalue()
                    if self.experiment.research is not None:
                        getattr(self.experiment.research.storage, 'experiments_'+name)[self.experiment.id] = content
                    else:
                        setattr(self, name+'_content', content)
                file.close()

    def _close_logger(self):
        """ Close experiment logger. """
        if len(self.logger.handlers) > 0:
            self.logger.removeHandler(self.logger.handlers[0])

class LocalExperimentStorage(BaseExperimentStorage):
    """ Experiment storage on disk.

    Parameters
    ----------
    experiment : Experiment

    loglevel : str, optional
        logging level, by default 'error'
    """
    def __init__(self, experiment, loglevel=None,  storage='local'):
        super().__init__(experiment, loglevel, storage)

        self.loglevel = loglevel or 'info'

        self._create_folder()
        self._create_logger()
        self._dump_config()
        self.create_redirection_streams()

    def close(self):
        """ Dump results and profiling and close created files. """
        self._dump_profile()
        self._close_files()
        self._close_logger()

    def dump_results(self, variable=None):
        """ Callable to dump results. The call is on the experiment side. """
        variables_to_dump = list(self.results.keys()) if variable is None else to_list(variable)
        for var in variables_to_dump:
            values = self.results[var]
            iteration = self.experiment.iteration
            variable_path = os.path.join(self.full_path, 'results', var)
            if not os.path.exists(variable_path):
                os.makedirs(variable_path)
            filename = os.path.join(variable_path, str(iteration))
            with open(filename, 'wb') as file:
                dill.dump(values, file)
            del self.results[var]

    # Initialization methods

    def _create_folder(self):
        """ Create folder for experiment results. """
        self.experiment_path = os.path.join('experiments', self.experiment.id)
        self.full_path = os.path.join(self.experiment.name, self.experiment_path)
        if not os.path.exists(self.full_path):
            os.makedirs(self.full_path)
        else:
            raise ValueError(f'Experiment folder {self.full_path} already exists.')

    def _create_logger(self):
        """ Create loggers. """
        if self.experiment.research:
            name = self.experiment.research.name
        else:
            name = self.experiment.executor.name
        logger_name = name + '.' + self.experiment.id
        path = os.path.join(name, 'experiments', self.experiment.id, 'experiment.log')
        self.logger = create_logger(logger_name, path, self.loglevel)

    def _dump_config(self):
        """ Dump config as serialized ConfigAlias instance and as a dict in JSON. """
        with open(os.path.join(self.full_path, 'config.dill'), 'wb') as file:
            dill.dump(self.experiment.config_alias, file)
        with open(os.path.join(self.full_path, 'config.json'), 'w') as file:
            json.dump(jsonify(self.experiment.config_alias.alias().config), file)

    def create_redirection_streams(self):
        """ Create streams to redirect stdout and stderr (if needed). """
        self.stdout_file = create_output_stream(
            self.experiment.redirect_stdout, True, 'stdout.txt', path=self.full_path, common=False
        )
        self.stderr_file = create_output_stream(
            self.experiment.redirect_stderr, True, 'stderr.txt', path=self.full_path, common=False
        )

    # Closing methods

    def _dump_profile(self):
        """ Dump profiling stats. """
        if self.profiler is not None:
            path = os.path.join(self.full_path, 'profiler.feather')
            self.profiler.profile_info.reset_index().to_feather(path)

    def _close_files(self):
        """ Close stdout/stderr files (if rederection was performed). """
        if not isinstance(self.stdout_file, (contextlib.nullcontext, type(None))):
            self.stdout_file.close()
        if not isinstance(self.stderr_file, (contextlib.nullcontext, type(None))):
            self.stderr_file.close()

    def _close_logger(self):
        """ Close experiment logger. """
        if len(self.logger.handlers) > 0:
            self.logger.removeHandler(self.logger.handlers[0])

class ClearMLExperimentStorage(BaseExperimentStorage):
    """ Experiment storage in ClearML.

    Parameters
    ----------
    experiment : Experiment

    loglevel : str, optional
        logging level, by default 'error'
    """
    def __init__(self, experiment, loglevel=None, storage='clearml'):
        from clearml import Task

        super().__init__(experiment, loglevel, storage)
        self.task = Task.init(
            project_name=self.name,
            task_name=self.experiment.id,
            # add_task_init_call=True,
        )
        self.task.connect_configuration(self.experiment.config.config)

        self._create_logger()
        self.create_redirection_streams()

    def close(self):
        """ Close task and profiler. """
        self.task.close()
        self._update_research_profiler()

    # Initialization methods

    def _create_logger(self):
        self.logger = ClearMLLogger(self.task.get_logger())

    def create_redirection_streams(self):
        """ Create streams to redirect stdout and stderr (if needed). """
        self.stdout_file = create_output_stream(True, False, common=False)
        self.stderr_file = create_output_stream(True, False, common=False)

    # Closing methods

    def dump_results(self, variable=None):
        """ Callable to dump results. """
        variables_to_dump = list(self.results.keys()) if variable is None else to_list(variable)
        for var in variables_to_dump:
            values = self.results[var]
            for iteration, value in values.items():
                self.logger.report_scalar('', variable, value, iteration)
            del self.results[var]

    def _update_research_profiler(self):
        """ Copy profiler stats to the common research storage. """
        experiment = self.experiment
        if experiment.research is not None and self.profiler is not None:
            experiment.research.profiler.put(experiment.id, self.profiler.profile_info)

class BaseResearchStorage:
    """ Storage for research data.

    Parameters
    ----------
    research : Research or Executor
        type depends on the instance which creates storage
    loglevel : str, optional
        logging level, by default 'error'
    storage : str, optional
        type of the storage, by default 'memory'
    """
    def __new__(cls, *args, storage='memory', **kwargs):
        _ = args, kwargs
        if storage == 'local':
            return super().__new__(LocalResearchStorage)
        if storage == 'memory':
            return super().__new__(MemoryResearchStorage)
        if storage == 'clearml':
            return super().__new__(ClearMLResearchStorage)

        raise ValueError(f'Unknown storage mode: {storage}')

    def __init__(self, research=None, loglevel=None, storage='memory'):
        _ = storage
        self.research = research
        self.loglevel = loglevel or 'error'
        self.logger = None

        self.results = None

        self.stdout_file = None
        self.stderr_file = None

    def close(self):
        """ Close shared memory managers and logger. """
        self.results.close_manager()
        self.profiler.close_manager()
        self._close_logger()

    def collect_env_state(self, env_meta_to_collect):
        """ Collector of environment state. """
        for item in env_meta_to_collect:
            args = item.pop('args', [])
            kwargs = item.pop('kwargs', {})
            self._collect_env_item(*args, **item, **kwargs)

    def _collect_env_item(self, cwd='.', dst=None, replace=None, commands=None, *args, **kwargs):
        """ Execute commands and save output. """
        if cwd == '.' and dst is None:
            dst = 'cwd'
        elif dst is None:
            dst = os.path.split(os.path.realpath(cwd))[1]

        if isinstance(commands, (tuple, list)):
            args = [*commands, *args]
        elif isinstance(commands, dict):
            kwargs = {**commands, **kwargs}

        all_commands = [('env_state', command) for command in args]
        all_commands = [*all_commands, *kwargs.items()]

        for filename, command in all_commands:
            if command.startswith('#'):
                if command[1:] == 'python':
                    result = sys.version
                else:
                    raise ValueError(f'Unknown env: {command}')
            else:
                process = subprocess.Popen(command.split(), stdout=subprocess.PIPE, cwd=cwd)
                output, _ = process.communicate()
                result = output.decode('utf')
            if replace is not None:
                for key, value in replace.items():
                    result = re.sub(key, value, result)

            self._store_env(result, dst, filename)

    def _store_env(self, *args, **kwargs):
        """ The method which defines the way to save the output of the command. """
        _ = args, kwargs
        raise AttributeError('`_store_env` method must be defined.')

    def close_files(self):
        """ Close stdout/stderr files (if rederection was performed). """
        for name in ['stdout', 'stderr']:
            file = getattr(self, name+'_file')
            if not isinstance(file, (contextlib.nullcontext, type(None))):
                file.close()

    def _close_logger(self):
        """ Close experiment logger. """
        if len(self.logger.handlers) > 0:
            self.logger.removeHandler(self.logger.handlers[0])

class MemoryResearchStorage(BaseResearchStorage):
    """ Research storage in RAM without any dumping on disk.

    Parameters
    ----------
    research : Research or Executor
        type depends on the instance which creates storage
    loglevel : str, optional
        logging level, by default 'error'
    """
    def __init__(self, research=None, loglevel=None, storage='memory'):
        super().__init__(research, storage)
        self.loglevel = loglevel or 'error'

        self._create_logger()

        self.results = ResearchResults(self.research.name, False)
        self.profiler = ResearchProfiler(self.research.name, self.research.profile)

        self._manager = mp.Manager()
        self.experiments_stdout = self._manager.dict()
        self.experiments_stderr = self._manager.dict()

        self._env = dict()

    def close(self):
        """ Close shared memory managers. """
        super().close()
        self.experiments_stdout = dict(self.experiments_stdout)
        self.experiments_stderr = dict(self.experiments_stderr)
        self._manager.shutdown()

    def _create_logger(self):
        self.logger = create_logger(self.research.name, None, self.loglevel)

    def create_redirection_streams(self):
        """ Create streams to redirect stdout and stderr (if needed). """
        self.stdout_file = create_output_stream(self.research.redirect_stdout, False, common=True)
        self.stderr_file = create_output_stream(self.research.redirect_stderr, False, common=True)

    def _store_env(self, result, dst, filename):
        key = os.path.join(dst, filename)
        self._env[key] = self._env.get(key, '') + result

    @property
    def env(self):
        return self._env

class LocalResearchStorage(BaseResearchStorage):
    """ Research storage on disk.

    Parameters
    ----------
    research : Research or Executor
        type depends on the instance which creates storage
    loglevel : str, optional
        logging level, by default 'error'
    """
    def __init__(self, research, loglevel, mode='w', storage='local'):
        super().__init__(research, storage)

        self.loglevel = loglevel or 'info'
        self.path = research.name
        if mode == 'w':
            self._create_folder()
            self._dump_research(research)
            self._create_logger()

        self.results = ResearchResults(self.research.name, True)
        self.profiler = ResearchProfiler(self.research.name, self.research.profile)

    def load(self):
        """ Load results and profiling stats. """
        self.results.load()
        self.profiler.load()

    def _create_folder(self):
        """ Create storage folder. """
        if os.path.exists(self.path):
            raise ValueError(f"Research storage '{self.path}' already exists")
        os.makedirs(self.path)
        for subfolder in ['env', 'experiments']:
            path = os.path.join(self.path, subfolder)
            if not os.path.exists(path):
                os.makedirs(path)

    def _dump_research(self, research):
        with open(os.path.join(self.path, 'research.dill'), 'wb') as f:
            dill.dump(research, f)
        with open(os.path.join(self.path, 'research.txt'), 'w') as f:
            f.write(str(research))

    def _create_logger(self):
        path = os.path.join(self.research.name, 'research.log')
        self.logger = create_logger(self.research.name, path, self.loglevel)

    def create_redirection_streams(self):
        """ Create streams to redirect stdout and stderr (if needed). """
        self.stdout_file = create_output_stream(
            self.research.redirect_stdout, True, 'stdout.txt', self.research.name, common=True
        )
        self.stderr_file = create_output_stream(
            self.research.redirect_stderr, True, 'stderr.txt', self.research.name, common=True
        )

    def _store_env(self, result, dst, filename):
        subfolder = os.path.join(self.path, 'env', dst)
        if not os.path.exists(subfolder):
            os.makedirs()
        with open(os.path.join(subfolder, filename + '.txt'), 'a') as file:
            file.write(result)

    @property
    def env(self):
        """ Environment state. """
        env = dict()
        filenames = glob.glob(os.path.join(self.path, 'env', '*'))
        for filename in filenames:
            name = os.path.splitext(os.path.basename(filename))[0]
            with open(filename, 'r') as file:
                env[name] = file.read().strip()
        return env

class ClearMLResearchStorage(BaseResearchStorage):
    """ Research storage in ClearML.

    Parameters
    ----------
    research : Research or Executor
        type depends on the instance which creates storage
    loglevel : str, optional
        logging level, by default 'error'
    """
    def __init__(self, research, loglevel, mode='w', storage='clearml'):
        from clearml import Task

        _ = loglevel, mode

        super().__init__(research, storage)
        self.task = Task.create(
            project_name=self.research.name,
            task_name='research',
            add_task_init_call=True,
        )

        self._create_logger()

    def close(self):
        # self.task.mark_completed()
        self.task.close()

    def _create_logger(self):
        self.logger = ClearMLLogger(self.task.get_logger())

    def create_redirection_streams(self):
        self.stdout_file = create_output_stream(True, False, common=True)
        self.stderr_file = create_output_stream(True, False, common=True)

    def collect_env_state(self, *args, **kwargs):
        _ = args, kwargs

class ClearMLLogger:
    """ Wrapper for ClearML Logger with 'info', 'debug', 'error' and 'critical' methods.

    Parameters
    ----------
    logger : clearml.Logger
    """
    def __init__(self, logger):
        self.logger = logger

    def __getattr__(self, name):
        if name in ['info', 'debug', 'error', 'critical']:
            loglevel = getattr(logging, name.upper())
            return lambda x: self.logger.report_text(x, level=loglevel, print_console=False)
        return getattr(self.logger, name)
