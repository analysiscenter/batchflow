""" Contains Variable class and Variables storage class """
import copy as cp
import threading
import logging

from .named_expr import eval_expr


class Variable:
    """ Pipeline variable """
    def __init__(self, name, default=None, lock=True, pipeline=None):
        self.name = name
        self.default = default
        self._lock = threading.Lock() if lock else None
        self.value = None
        self.initialize(pipeline=pipeline)

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_lock'] = state['_lock'] is not None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._lock = threading.Lock() if state['_lock'] else None

    def get(self):
        """ Return a variable value """
        return self.value

    def set(self, value):
        """ Assign a variable value """
        self.lock()
        self.value = value
        self.unlock()

    def initialize(self, pipeline=None):
        """ Initialize a variable value """
        value = eval_expr(self.default, pipeline=pipeline)
        self.set(value)

    def lock(self):
        """ Acquire lock """
        if self._lock:
            self._lock.acquire()

    def unlock(self):
        """ Release lock """
        if self._lock:
            self._lock.release()


class VariableDirectory:
    """ Storage for pipeline variables """
    def __init__(self, strict=False):
        self.variables = {}
        self.strict = strict
        self._lock = threading.Lock()

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('_lock')
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._lock = threading.Lock()

    def lock(self, name=None):
        """ Lock the directory itself or a variable """
        if name is None:
            if self._lock:
                self._lock.acquire()
        else:
            self.variables[name].lock()

    def unlock(self, name=None):
        """ Unlock the directory itself or a variable """
        if name is None:
            if self._lock:
                self._lock.release()
        else:
            self.variables[name].unlock()

    def copy(self):
        return cp.deepcopy(self)

    def __copy__(self):
        """ Make a shallow copy of the directory """
        new_dir = VariableDirectory()
        new_dir.strict = self.strict
        new_dir.variables = {**self.variables}
        return new_dir

    def __deepcopy__(self, memo):
        """ Make a deep copy of the directory """
        _ = memo
        new_dir = VariableDirectory()
        new_dir.strict = self.strict
        new_dir.create_many(self)
        return new_dir

    def __add__(self, other):
        if not isinstance(other, VariableDirectory):
            raise TypeError(f"VariableDirectory is expected, but given '{type(other).__name__}'")

        new_dir = self.copy()
        new_dir.create_many(other)
        return new_dir

    def items(self):
        """ Return a sequence of (name, params) for all variables """
        for v in self.variables:
            var = self.variables[v].__getstate__()
            var.pop('value')
            var['lock'] = var['_lock']
            var.pop('_lock')
            yield v, var

    def exists(self, name):
        """ Checks if a variable already exists """
        return name in self.variables

    def create(self, name, *args, pipeline=None, **kwargs):
        """ Create a variable """
        if not self.exists(name):
            with self._lock:
                if not self.exists(name):
                    self.variables[name] = Variable(name, *args, pipeline=pipeline, **kwargs)

    def create_many(self, variables, pipeline=None):
        """ Create many variables at once """
        if isinstance(variables, (tuple, list)):
            variables = dict(zip(variables, [None] * len(variables)))

        for name, var in variables.items():
            var = var or {}
            var.pop('name', '')
            var.pop('args', ())
            kwargs = var.pop('kwargs', {})
            self.create(name, **var, **kwargs, pipeline=pipeline)

    def initialize(self, pipeline=None):
        """ Initialize all variables """
        with self._lock:
            for v in self.variables:
                self.variables[v].initialize(pipeline=pipeline)

    def _should_create(self, name, create=False):
        create = not self.strict or create
        if not self.exists(name):
            if create:
                logging.warning("Variable '%s' has not been initialized", name)
            else:
                raise KeyError(f"Variable '{name}' does not exists")
        return create

    def get(self, name, *args, create=False, pipeline=None, **kwargs):
        """ Return a variable value """
        if self._should_create(name, create or len(args) + len(kwargs) > 0):
            self.create(name, *args, pipeline=pipeline, **kwargs)
        var = self.variables[name].get()
        return var

    def set(self, name, value, pipeline=None):
        """ Set a variable value """
        if self._should_create(name):
            self.create(name, pipeline=pipeline)
        self.variables[name].set(value)

    def delete(self, name):
        """ Remove the variable with a given name """
        if not self.exists(name):
            logging.warning("Variable '%s' does not exist", name)
        else:
            self.variables.pop(name)
