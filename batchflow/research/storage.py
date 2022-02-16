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

class ExperimentStorage:
    def __new__(cls, *args, storage='memory', **kwargs):
        _ = args, kwargs
        if storage == 'local':
            return super().__new__(LocalExperimentStorage)
        if storage == 'memory':
            return super().__new__(MemoryExperimentStorage)
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

        self.create_profiler()

    def create_profiler(self):
        profile = self.experiment.profile
        if profile == 2 or isinstance(profile, str) and 'detailed'.startswith(profile):
            self.profiler = ExperimentProfiler(detailed=True)
        elif profile == 1 or profile is True:
            self.profiler = ExperimentProfiler(detailed=False)
        else: # 0, False, None
            self.profiler = None

    def update_variable(self, name, value): #iteration is availiable from experiment
        results = self.results.get(name, OrderedDict())
        results[self.experiment.iteration] = value
        self.results[name] = results #TODO: do we need it?

    def copy_results_to_research_storage(self):
        experiment = self.experiment
        if experiment.research is not None:
            experiment.research.results.put(experiment.id, experiment.results, experiment.config_alias)
            if self.profiler is not None:
                experiment.research.profiler.put(experiment.id, self.profiler.profile_info)

    def close_logger(self):
        """ Close experiment logger. """
        self.logger.removeHandler(self.logger.handlers[0])

class MemoryExperimentStorage(ExperimentStorage):
    def __init__(self, experiment, loglevel=None, storage='memory'):
        super().__init__(experiment, loglevel, storage)

        self.results = OrderedDict()

        self.create_logger()
        self.create_redirection_files()

    def create_logger(self):
        if self.experiment.research:
            name = self.experiment.research.name
        else:
            name = self.experiment.executor.name
        logger_name = name + '.' + self.experiment.id
        self.logger = create_logger(logger_name, None, self.loglevel)

    def create_redirection_files(self):
        self.stdout_file = create_output_stream(self.experiment.redirect_stdout, False, common=False)
        self.stderr_file = create_output_stream(self.experiment.redirect_stderr, False, common=False)

    def close(self):
        self.copy_results_to_research_storage()
        self.close_files()
        self.close_logger()

    def close_files(self):
        """ Close stdout/stderr files (if rederection was performed). """
        for name in ['stdout', 'stderr']:
            file = getattr(self, name+'_file')
            if not isinstance(file, (contextlib.nullcontext, type(None))):
                if isinstance(file, io.StringIO):
                    content = file.getvalue()
                    if self.experiment.research is not None:
                        getattr(self.experiment.research.storage, 'experiments_'+name)[self.experiment.id] = content
                    else:
                        setattr(self, name+'_content', content)
                file.close()

class LocalExperimentStorage(ExperimentStorage):
    def __init__(self, experiment, loglevel=None,  storage='local'):
        super().__init__(experiment, loglevel, storage)

        self.loglevel = loglevel or 'info'

        self.create_folder()
        self.dump_config()
        self.create_logger()
        self.create_empty_results()
        self.create_redirection_files()

    def create_folder(self):
        """ Create folder for experiment results. """
        self.experiment_path = os.path.join('experiments', self.experiment.id)
        self.full_path = os.path.join(self.experiment.name, self.experiment_path)
        if not os.path.exists(self.full_path):
            os.makedirs(self.full_path)
        else:
            raise ValueError(f'Experiment folder {self.full_path} already exists.')

    def dump_config(self):
        """ Dump config (as serialized ConfigAlias instance). """
        with open(os.path.join(self.full_path, 'config.dill'), 'wb') as file:
            dill.dump(self.experiment.config_alias, file)
        with open(os.path.join(self.full_path, 'config.json'), 'w') as file:
            json.dump(jsonify(self.experiment.config_alias.alias().config), file)

    def create_logger(self):
        if self.experiment.research:
            name = self.experiment.research.name
        else:
            name = self.experiment.executor.name
        logger_name = name + '.' + self.experiment.id
        path = os.path.join(name, 'experiments', self.experiment.id, 'experiment.log')
        self.logger = create_logger(logger_name, path, self.loglevel)

    def create_empty_results(self):
        self.results = OrderedDict()

    def create_redirection_files(self):
        self.stdout_file = create_output_stream(
            self.experiment.redirect_stdout, True, 'stdout.txt', path=self.full_path, common=False
        )
        self.stderr_file = create_output_stream(
            self.experiment.redirect_stderr, True, 'stderr.txt', path=self.full_path, common=False
        )

    def close(self):
        self.copy_results_to_research_storage()
        self.dump_results()
        self.dump_profile()
        self.close_files()
        self.close_logger()

    def dump_results(self, variable=None):
        """ Callable to dump results. """
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

    def dump_profile(self):
        if self.profiler is not None:
            path = os.path.join(self.full_path, 'profiler.feather')
            self.profiler.profile_info.reset_index().to_feather(path)

    def close_files(self):
        """ Close stdout/stderr files (if rederection was performed). """
        if not isinstance(self.stdout_file, (contextlib.nullcontext, type(None))):
            self.stdout_file.close()
        if not isinstance(self.stderr_file, (contextlib.nullcontext, type(None))):
            self.stderr_file.close()

class ResearchStorage:
    def __new__(cls, *args, storage='memory', **kwargs):
        _ = args, kwargs
        if storage == 'local':
            return super().__new__(LocalResearchStorage)
        if storage == 'memory':
            return super().__new__(MemoryResearchStorage)
        raise ValueError(f'Unknown storage mode: {storage}')

    def __init__(self, research=None, loglevel=None, storage='memory'):
        _ = storage
        self.research = research
        self.loglevel = loglevel or 'error'
        self.logger = None

        self.results = None
        self.profiler = ResearchProfiler(self.research.name, self.research.profile)

        self.stdout_file = None
        self.stderr_file = None

        self._manager = mp.Manager()
        self.experiments_stdout = self._manager.dict()
        self.experiments_stderr = self._manager.dict()

    def collect_env_state(self, env_meta_to_collect):
        for item in env_meta_to_collect:
            args = item.pop('args', [])
            kwargs = item.pop('kwargs', {})
            self._collect_env_state(*args, **item, **kwargs)

    def _collect_env_state(self, cwd='.', dst=None, replace=None, commands=None, *args, **kwargs):
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

            self.store_env(result, dst, filename)

    def create_logger(self):
        name = self.research.name
        path = os.path.join(name, 'research.log')
        self.logger = create_logger(name, path, self.loglevel)

    def close_files(self):
        """ Close stdout/stderr files (if rederection was performed). """
        for name in ['stdout', 'stderr']:
            file = getattr(self, name+'_file')
            if not isinstance(file, (contextlib.nullcontext, type(None))):
                file.close()

    def close(self):
        self.results.close_manager()
        self.profiler.close_manager()
        self.experiments_stdout = dict(self.experiments_stdout)
        self.experiments_stderr = dict(self.experiments_stderr)
        self._manager.shutdown()

class MemoryResearchStorage(ResearchStorage):
    def __init__(self, research=None, loglevel=None, storage='memory'):
        super().__init__(research, storage)
        self.loglevel = loglevel or 'error'

        self.create_logger()
        self.create_redirection_files()
        self.results = ResearchResults(self.research.name, False)

        self._env = dict()

    def create_logger(self):
        self.logger = create_logger(self.research.name, None, self.loglevel)

    def store_env(self, result, dst, filename):
        key = os.path.join(dst, filename)
        self._env[key] = self._env.get(key, '') + result

    @property
    def env(self):
        return self._env

    def create_redirection_files(self):
        self.stdout_file = create_output_stream(self.research.redirect_stdout, False, common=True)
        self.stderr_file = create_output_stream(self.research.redirect_stderr, False, common=True)

class LocalResearchStorage(ResearchStorage):
    def __init__(self, research, loglevel, mode='w', storage='local'):
        super().__init__(research, storage)

        self.loglevel = loglevel or 'info'
        self.path = research.name
        if mode == 'w':
            self.create_folder()
        self.dump_research(research)
        self.create_logger()
        self.create_redirection_files()
        self.results = ResearchResults(self.research.name, True)

    def create_folder(self):
        """ Create storage folder. """
        if os.path.exists(self.path):
            raise ValueError(f"Research storage '{self.path}' already exists")
        os.makedirs(self.path)
        for subfolder in ['env', 'experiments']:
            path = os.path.join(self.path, subfolder)
            if not os.path.exists(path):
                os.makedirs(path)

    def dump_research(self, research):
        with open(os.path.join(self.path, 'research.dill'), 'wb') as f:
            dill.dump(research, f)
        with open(os.path.join(self.path, 'research.txt'), 'w') as f:
            f.write(str(research))

    def create_logger(self):
        path = os.path.join(self.research.name, 'research.log')
        self.logger = create_logger(self.research.name, path, self.loglevel)

    def store_env(self, result, dst, filename):
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

    def create_redirection_files(self):
        self.stdout_file = create_output_stream(
            self.research.redirect_stdout, True, 'stdout.txt', self.research.name, common=True
        )
        self.stderr_file = create_output_stream(
            self.research.redirect_stderr, True, 'stderr.txt', self.research.name, common=True
        )

    def load(self):
        self.results = ResearchResults(self.research.name, self.research.dump_results)
        self.profiler = ResearchProfiler(self.research.name, self.research.profile)
        self.results.load()
        self.profiler.load()
