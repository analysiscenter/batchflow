import os
import sys
import re
import dill
import glob
import logging
import subprocess

from .utils import create_logger

class ResearchStorage:
    def __new__(cls, dump_results, *args, **kwargs):
        if dump_results:
            return LocalResearchStorage(*args, **kwargs)
        else:
            return MemoryResearchStorage(*args, **kwargs)

    def __init__(self, research):
        self.research = research

    def create_logger(self):
        pass

    def _get_env_state(self, cwd='.', dst=None, replace=None, commands=None, *args, **kwargs):
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

class MemoryResearchStorage(ResearchStorage):
    def __init__(self, research):
        super().__init__(research)
        self._env = dict()

    def create_logger(self, name, loglevel=None):
        loglevel = 'error' or loglevel
        return create_logger(name, None, loglevel)

    def store_env(self, result, dst, filename):
        key = os.path.join(dst, filename)
        self._env[key] = self._env.get(key, '') + result

    @property
    def env(self):
        return self._env

class LocalResearchStorage(ResearchStorage):
    def __init__(self, research, path):
        super().__init__(research)

        self.path = path
        self.create_folder()
        self.dump_research(research)

    def create_folder(self):
        """ Create storage folder. """
        if os.path.exists(path):
            raise ValueError(f"Research storage '{path}' already exists")
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

    def create_logger(self, name, loglevel=None):
        loglevel = 'info' or loglevel
        return create_logger(name, self.path, loglevel)

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

class ClearMLResearchStorage(ResearchStorage):
    def __init__(self, research, path):
        super().__init__(research)
