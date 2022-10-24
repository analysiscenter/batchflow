""" Contains pipeline class """
# pylint:disable=undefined-variable
import sys
import time
from functools import partial
import threading
import asyncio
import logging
import warnings

from .base import Baseset
from .config import Config
from .batch import Batch
from .decorators import deprecated
from .exceptions import StopPipeline
from .named_expr import NamedExpression, V, eval_expr
from .once_pipeline import OncePipeline
from .model_dir import ModelDirectory
from .variables import VariableDirectory
from .models.metrics import (ClassificationMetrics, SegmentationMetricsByPixels,
                             SegmentationMetricsByInstances, RegressionMetrics, Loss)

from ._const import *       # pylint:disable=wildcard-import
from .utils import save_data_to
from .utils_random import make_rng
from .pipeline_executor import PipelineExecutor
from .profiler import PipelineProfiler


METRICS = dict(
    classification=ClassificationMetrics,
    segmentation=SegmentationMetricsByPixels,
    mask=SegmentationMetricsByPixels,
    instance=SegmentationMetricsByInstances,
    regression=RegressionMetrics,
    loss=Loss,
)


def mult_option(a, b):
    """ Multiply even if any arg is None """
    return a * b if a is not None and b is not None else a if a is not None else b


def hashable(x):
    """ Check if x is hashable """
    try:
        hash(x)
    except TypeError:
        return False
    return True


class Pipeline:
    """ Pipeline """
    def __init__(self, dataset=None, config=None, pipeline=None, actions=None, strict=False, proba=None, repeat=None):
        # pylint: disable=protected-access

        if pipeline is None:
            self.dataset = dataset
            self.config = config or {}
            self._actions = actions or []
            self._lazy_run = None
            self.models = ModelDirectory()
            self.variables = VariableDirectory(strict)
            self.strict = strict
            self.before = OncePipeline(self)
            self.after = OncePipeline(self)
            self._namespaces = []
        else:
            self.dataset = pipeline.dataset
            config = config or {}
            _config = pipeline.config or {}
            self.config = {**config, **_config}
            self._actions = actions or pipeline._actions[:]
            if self.num_actions == 1:
                if proba is not None:
                    if self.get_last_action_repeat() is None:
                        self._actions[-1]['proba'] = mult_option(proba, self.get_last_action_proba())
                elif repeat is not None:
                    if self.get_last_action_proba() is None:
                        self._actions[-1]['repeat'] = mult_option(repeat, self.get_last_action_repeat())
            self._lazy_run = pipeline._lazy_run
            self.variables = pipeline.variables.copy()
            self.strict = pipeline.strict
            self.models = pipeline.models.copy()
            self._namespaces = pipeline._namespaces
            self.before = pipeline.before.copy()
            self.before.pipeline = self
            self.after = pipeline.after.copy()
            self.after.pipeline = self

        self._dataset = None
        self.config = Config(self.config)

        self._batch_generator = None
        self.iter_params = Baseset.get_default_iter_params()
        self._rest_batch = None
        self.variables_initialised = False
        self._local = threading.local()
        self.random = None
        self.random_seed = None

        self._profiler = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state['random'] = getattr(self._local, 'random', None)
        state.pop('_local')
        state['_profiler'] = None
        state['_batch_generator'] = None
        return state

    def __setstate__(self, state):
        self._local = threading.local()
        for k, v in state.items():
            setattr(self, k, v)

    def __enter__(self):
        """ Create a context and return an empty pipeline non-bound to any dataset """
        return type(self)()

    def __exit__(self, exc_type, exc_value, trback):
        pass

    @property
    def random(self):
        return self._local.random if hasattr(self._local, 'random') else None

    @random.setter
    def random(self, value):
        self._local.random = value

    @property
    def random_seed(self):
        return self._local.random_seed if hasattr(self._local, 'random_seed') else None

    @random_seed.setter
    def random_seed(self, value):
        self._local.random_seed = value
        self.random = make_rng(value)

    @classmethod
    def from_pipeline(cls, pipeline, actions=None, proba=None, repeat=None):
        """ Create a pipeline from another pipeline """
        if proba is None:
            if repeat is None:
                new_p = cls(pipeline=pipeline, actions=actions)
            else:
                if pipeline.num_actions == 1 and pipeline.get_last_action_proba() is None:
                    new_p = cls(pipeline=pipeline, repeat=repeat)
                else:
                    new_p = cls()
                    new_p.append_pipeline(pipeline, repeat=repeat)
        else:
            if pipeline.num_actions == 1 and pipeline.get_last_action_repeat() is None:
                new_p = cls(pipeline=pipeline, proba=proba)
            else:
                new_p = cls()
                new_p.append_pipeline(pipeline, proba=proba)
        return new_p

    @classmethod
    def concat(cls, pipe1, pipe2):
        """ Create a new pipeline concatenating two given pipelines """
        # pylint: disable=protected-access
        new_p1 = cls.from_pipeline(pipe1)
        new_p1._actions += pipe2._actions[:]
        new_p1.config.update(pipe2.config)
        new_p1.variables += pipe2.variables
        new_p1.models += pipe2.models
        if new_p1.dataset is None:
            new_p1.dataset = pipe2.dataset
        new_p1._lazy_run = new_p1._lazy_run or pipe2._lazy_run
        new_p1.before = pipe1.before.concat(pipe1.before, pipe2.before)
        new_p1.before.pipeline = new_p1
        new_p1.after = pipe1.after.concat(pipe1.after, pipe2.after)
        new_p1.after.pipeline = new_p1
        return new_p1

    def get_last_action_proba(self):
        """ Return a probability of the last action """
        return self._actions[-1]['proba']

    def get_last_action_repeat(self):
        """ Return a repeat count of the last action """
        return self._actions[-1]['repeat']

    def __add__(self, other):
        if isinstance(other, OncePipeline):
            other = other.pipeline
        if not isinstance(other, Pipeline):
            raise TypeError("Both operands should be Pipelines")
        if len(other._actions) > 0 and other._actions[0]['name'] == REBATCH_ID:
            new_p = self.from_pipeline(other)
            new_p._actions[0]['pipeline'] = self + new_p._actions[0]['pipeline']
            return new_p
        return self.concat(self, other)

    def __matmul__(self, other):
        if self.num_actions == 0:
            raise ValueError("Cannot add probability to an empty pipeline")
        if isinstance(other, NamedExpression):
            pass
        elif not isinstance(other, float) and other not in [0, 1]:
            raise TypeError("Probability should be float or 0 or 1")
        else:
            other = float(other) if int(other) != 1 else None
        return self.from_pipeline(self, proba=other)

    def __mul__(self, other):
        if isinstance(other, int) and other < 0:
            raise ValueError("Repeat count cannot be negative. Use as pipeline * positive_number")
        if isinstance(other, float):
            raise ValueError("Repeat count cannot be float. Use as pipeline * integer")
        new_p = self.from_pipeline(self, repeat=other)
        return new_p

    def __lshift__(self, other):
        new_p = self.from_pipeline(self)
        if isinstance(other, (Baseset, NamedExpression)):
            new_p.dataset = other
            return new_p
        if isinstance(other, (Config, dict)):
            new_p.set_config(other)
            return new_p
        raise TypeError("Pipeline might take only Dataset or Config. Use as pipeline << dataset or pipeine << config")

    def _is_batch_method(self, name, namespace=Batch):
        if self._dataset is not None:
            namespace = namespace or self._dataset.batch_class
        if hasattr(namespace, name) and callable(getattr(namespace, name)):
            return True
        return any(self._is_batch_method(name, subcls) for subcls in namespace.__subclasses__())

    def get_action_name(self, action, add_index=False):
        """ Return a pretty action name """
        if action['name'] == '#_from_ns':
            name = action['method'].__name__
        if action['name'].startswith('#_'):
            name = action['name'][2:]
        else:
            name = action['name']
        idx = self._actions.index(action)
        return name if add_index is False else f'{name} #{idx}'

    def add_namespace(self, *namespaces):
        """ Add namespace to call pipeline actions from

        Parameters
        ----------
        namespace : str or module
            a module name as a string or a module reference

        Examples
        --------

        Add a standard time module and numpy::

            dataset.p.add_namespace('time', numpy)

        Notes
        -----
        The current module and dataset module are included by default.

        Passing a namespace as a string is necessary for multiprocess executions,
        e.g. when running a pipeline with prefetch and mpc target.
        """
        self._namespaces.extend(namespaces)
        return self

    @property
    def _all_namespaces(self):
        common_namespaces = [sys.modules["__main__"]]
        if isinstance(self.dataset, NamedExpression):
            if self._dataset is not None:
                common_namespaces.append(self._dataset)
        else:
            common_namespaces.append(self.dataset)

        namespaces = [sys.modules[namespace] if isinstance(namespace, str) else namespace \
                      for namespace in self._namespaces]

        return common_namespaces + namespaces

    def is_method_from_ns(self, name):
        return any(hasattr(namespace, name) for namespace in self._all_namespaces)

    def get_method(self, name):
        """ Return a method by the name """
        for namespace in self._all_namespaces:
            if hasattr(namespace, name):
                return getattr(namespace, name)
        return None

    def __getattr__(self, name):
        """ Check if an unknown attr is an action from some batch class """
        if name[:2] == '__' and name[-2:] == '__':
            # if a magic method is not defined, throw an error
            raise AttributeError(f'Unknown magic method: {name}')
        if self._is_batch_method(name):
            return partial(self._add_action, name)
        if self.is_method_from_ns(name):
            return partial(self._add_action, CALL_FROM_NS_ID, _name=name)
        raise AttributeError(f"{name} not found in class {self.__class__.__name__}")

    @property
    def num_actions(self):
        """ Return index length """
        return len(self._actions)

    def _add_action(self, name, *args, _name=None, _args=None, **kwargs):
        """ Add new action to the log of future actions """
        actions = self._actions.copy()
        if name == CALL_FROM_NS_ID:
            method = self.get_method(_name)
            save_to = kwargs.pop('save_to', None)
            actions.append({'name': name, 'args': args, 'kwargs': kwargs,
                            'method': method, 'save_to': save_to,
                            'proba': None, 'repeat': None})
        else:
            action = {'name': name, 'args': args, 'kwargs': kwargs, 'proba': None, 'repeat': None}
            if _args:
                action.update(**_args)
            actions.append(action)
        new_p = self.from_pipeline(self, actions=actions)
        return new_p

    def append_pipeline(self, pipeline, proba=None, repeat=None):
        """ Add a nested pipeline to the log of future actions """
        self._actions.append({'name': PIPELINE_ID, 'pipeline': pipeline, 'proba': proba, 'repeat': repeat})


    LINE_LENGTH = 80
    ITEM_LENGTH = 40
    PREFIX = '\033[1m\033[4m'

    def __str__(self):
        """ Textual representation of a pipeline: list all actions and their keyword parameters. """
        # TODO: with the help of `inspect` and `get_method` method, can match the `args` to their actual names
        msg = []
        for i, action in enumerate(self._actions):
            name = self.get_action_name(action)
            indent = (len(name) + len(str(i)) + 3) * ' '

            # Combine kwargs passed directly to actions and ones from pipeline methods
            kwargs = {**action['kwargs'], **action}
            for key in ['name', 'args', 'kwargs', 'proba', 'repeat']:
                kwargs.pop(key)

            # Make string with parameters
            lines_kwargs = []
            str_kwargs = ''
            for key, value in kwargs.items():
                # Control the length of one parameter value
                value_ = repr(value)
                value_ = value_ if len(value_) < self.ITEM_LENGTH else '[...]'
                str_kwargs += f'{key}={value_}, '

                # Control the length of current line
                if len(str_kwargs) + len(indent) > self.LINE_LENGTH:
                    lines_kwargs.append(str_kwargs[:-2])
                    str_kwargs = ''
            if str_kwargs:
                lines_kwargs.append(str_kwargs[:-2])
            str_kwargs = f'\n{indent}'.join(lines_kwargs)

            action_msg = f'#{i} {self.PREFIX}{name}\033[0m({str_kwargs})'
            msg.append(action_msg)

        return '\n'.join(msg)


    @property
    def index(self):
        """ Return index of the source dataset """
        return self._dataset.index

    @property
    def indices(self):
        """ Return the sequence of indices of the source dataset """
        return self.index.indices

    def __len__(self):
        """ Return index length """
        return len(self.index)

    def set_config(self, config, clear=False):
        """ Update pipeline's config

        Parameters
        ----------
        config: dict
            configuration parameters
        clear : bool
            whether to clear the current config
        """
        if clear:
            self.config = {}
        self.config.update(config)
        return self

    def update_config(self, config):
        """ Update pipeline's config

        Parameters
        ----------
        config: dict
            configuration parameters
        clear : bool
            whether to clear the current config
        """
        return self.set_config(config, clear=False)


    def set_dataset(self, dataset):
        """ Link the pipeline to a dataset

        Parameters
        ----------
        dataset : Dataset
            a dataset to link to

        Notes
        -----
        This method is a declarative version of ``pipeline << dataset``,
        so it is executed only when the pipeline is run.

        It is always run as the first action in the pipeline chain despite it's actual location.
        """
        if self.dataset is not None:
            logging.warning("Initial dataset will be changed.")
        self.dataset = dataset
        return self

    def has_variable(self, name):
        """ Check if a variable exists

        Parameters
        ----------
        name : str
            a name of the variable

        Returns
        -------
        True if the variable exists
        """
        return hashable(name) and self.variables.exists(name)

    def get_variable(self, name, *args, create=False, **kwargs):
        """ Return a variable value.

        If the variable does not exists, it might be created and initialized (see `init_variable` below)

        Parameters
        ----------
        name : string
            a name of the variable
        create : bool
            whether to create a variable if it does not exist. Default is `False`.
        args, kwargs
            parameters for :meth:`~.Pipeline.init_variable` if ``create`` is True.

        Returns
        -------
        a value of the variable

        Raises
        ------
        `KeyError` if a variable does not exist
        """
        return self.variables.get(name, *args, create=create, pipeline=self, **kwargs)

    def v(self, name, *args, **kwargs):
        """ A shorter alias for get_variable() """
        return self.get_variable(name, *args, **kwargs)

    def init_lock(self, name='lock', **kwargs):
        """ Create a lock as a pipeline variable

        Parameters
        ----------
        name : string
            a lock name

        Returns
        -------
        self - in order to use it in the pipeline chains

        Examples
        --------
        >>> pp = dataset.p
                    .init_lock("model_update")
        """
        self.before.init_lock(name, **kwargs)
        return self

    def acquire_lock(self, name='lock', **kwargs):
        """ Acquire lock

        Parameters
        ----------
        name : string
            a lock name

        Returns
        -------
        self - in order to use it in the pipeline chains
        """
        return self._add_action(ACQUIRE_LOCK_ID, _args=dict(lock_name=name), **kwargs)

    def _exec_acquire_lock(self, batch, action):
        if not batch.pipeline.has_variable(action['lock_name']):
            self.init_lock(action['lock_name'])
        batch.pipeline.v(action['lock_name']).acquire(**action['kwargs'])

    def release_lock(self, name='lock', **kwargs):
        """ Release lock

        Parameters
        ----------
        name : string
            a lock name

        Returns
        -------
        self - in order to use it in the pipeline chains
        """
        return self._add_action(RELEASE_LOCK_ID, _args=dict(lock_name=name), **kwargs)

    def _exec_release_lock(self, batch, action):
        batch.pipeline.v(action['lock_name']).release(**action['kwargs'])

    def discard_batch(self):
        """ Discard the batch
        (helpful in multiprocessing prefetching to prevent passing the batch back)

        Returns
        -------
        self - in order to use it in the pipeline chains
        """
        return self._add_action(DISCARD_BATCH_ID)

    def init_variable(self, name, default=None, lock=True, **kwargs):
        """ Create a variable if not exists.
        If the variable exists, does nothing.

        Parameters
        ----------
        name : string
            a name of the variable
        default
            an initial value for the variable set when pipeline is created
        lock : bool
            whether to lock a variable before each update (default: True)

        Returns
        -------
        self - in order to use it in the pipeline chains

        Examples
        --------
        >>> pp = dataset.p.
                    .init_variable("iterations", default=0)
                    .init_variable("accuracy", 0)
                    .init_variable("loss_history", [])
                    .load('/some/path', fmt='blosc')
                    .train_resnet()
        """
        self.before.init_variable(name, default, lock, **kwargs)
        return self

    def init_variables(self, *variables):
        """ Create several variables

        Parameters
        ----------
        variables : dict or tuple
            if tuple, contains variable names which will have None as default values
            if dict, then mapping from variable names to values and init params (see :meth:`~.Pipeline.init_variable`)

        Returns
        -------
        self - in order to use it in the pipeline chains

        Examples
        --------
        >>> pp = dataset.p
                    .init_variables({"loss_history": dict(default=[]),
                                     "predictions", dict(default=[])})
                    .init_variables("metrics", "counter", "worst_prediction")
                    .load('/some/path', fmt='blosc')
                    .train_resnet()
        """
        if len(variables) == 1:
            variables = variables[0]
        self.variables.create_many(variables)
        return self

    def _init_all_variables(self):
        self.variables.initialize(pipeline=self)

    def set_variable(self, name, value, mode='w', batch=None):
        """ Set a variable value
        If the variable does not exists, it will be created, however, the warning will be displayed that
        the variable was not initialized.

        Parameters
        ----------
        name : str or a named expression - a variable name

        value
            an updating value, could be a value of any type or a named expression

        mode : str
            a method to update a variable value, could be one of:

            - 'w' or 'write' to rewrite a variable with a new value. This is a default mode.
            - 'a' or 'append' to append a value to a variable (e.g. if a variable is a list).
            - 'e' or 'extend' to extend a variable with a new value (e.g. if a variable is a list).
            - 'u' or 'update' to update a variable with a new value (e.g. if a variable is a dict).

            For sets and dicts 'a' and 'u' do exactly the same.

        Notes
        -----
        Unlike :meth:`~.Pipeline.update_variable` this method sets a new value immediately.
        So ``set_variable`` is imperative and may be used within actions, while ``update_variable``
        is declarative and should be used in pipeline definition chains.
        """
        V(name, mode=mode).set(value, batch=batch, pipeline=self)

    def assign_variable(self, name, value):
        """ Assign a value to a variable """
        self.variables.set(name, value, pipeline=self)

    def delete_variable(self, name):
        """ Delete a variable
        If the variable does not exists, the warning will be issued.

        Parameters
        ----------
        name : str
            a name of the variable

        Returns
        -------
        self - in order to use it in the pipeline chains
        """
        self.variables.delete(name)
        return self

    def del_variable(self, name):
        """ Delete a variable
        Same as `delete_variable(name)`
        """
        return self.delete_variable(name)

    def delete_all_variables(self):
        """ Delete all variables """
        self.variables = VariableDirectory()

    def save_to(self, dst, value=None):
        """ Save a value of a given named expression lazily during pipeline execution

        Parameters
        ----------
        dst : NamedExpression or any data container
            destination

        value
            an updating value, could be a value of any type or a named expression

        Returns
        -------
        self - in order to use it in the pipeline chains

        Notes
        -----
        This method does not change a value of the variable until the pipeline is run.
        So it should be used in pipeline definition chains only.
        :func:`~.save_data_to` is imperative and may be used to change variable value within actions.
        """
        return self.update(dst, value)

    def update(self, expr, value=None):
        """ Update a value of a given named expression lazily during pipeline execution

        Parameters
        ----------
        expr : NamedExpression
            an expression

        value
            an updating value, could be a value of any type or a named expression

        Returns
        -------
        self - in order to use it in the pipeline chains

        Notes
        -----
        This method does not change a value of the variable until the pipeline is run.
        So it should be used in pipeline definition chains only.
        ``set_variable`` is imperative and may be used to change variable value within actions.
        """
        return self._add_action(UPDATE_ID, _args=dict(expr=expr, value=value))

    def _exec_update(self, batch, action):
        action['expr'].set(action['value'], batch=batch)


    @deprecated("update_variable() is deprecated. Use pipeline.update(V(name), value) instead.")
    def update_variable(self, name, value=None, mode='w'):
        """ Update a value of a given variable lazily during pipeline execution

        Parameters
        ----------
        name : str or a named expression - a variable name

        value
            an updating value, could be a value of any type or a named expression

        mode : str
            a method to update a variable value, could be one of:

            - 'w' or 'write' to rewrite a variable with a new value. This is a default mode.
            - 'a' or 'append' to append a value to a variable (e.g. if a variable is a list).
            - 'e' or 'extend' to extend a variable with a new value (e.g. if a variable is a list).
            - 'u' or 'update' to update a variable with a new value (e.g. if a variable is a dict).

            For sets and dicts 'a' and 'u' do exactly the same.

        Returns
        -------
        self - in order to use it in the pipeline chains

        Notes
        -----
        Unlike :meth:`~.Pipeline.set_variable` this method does not change a value of the variable
        until the pipeline is run. So it should be used in pipeline definition chains only.
        ``set_variable`` is imperative and may be used to change variable value within actions.
        """
        return self._add_action(UPDATE_VARIABLE_ID, _args=dict(var_name=name, value=value, mode=mode))

    def _exec_update_variable(self, batch, action):
        self.set_variable(action['var_name'], action['value'], action['mode'], batch=batch)

    def print(self, *args, **kwargs):
        """ Print a value during pipeline execution """
        return self._add_action(PRINT_ID, *args, **kwargs)

    def _exec_print(self, _, action):
        args_value = action['args']
        kwargs_value = action['kwargs']

        args = []
        if len(args_value) == 0:
            pass
        else:
            args.extend(args_value)
        if len(kwargs_value) == 0:
            pass
        else:
            for k in kwargs_value:
                args.append(str(k) + '=' + str(kwargs_value[k]))
        try:
            print(*args)
        except OSError:
            pass

    def call(self, fn, *args, save_to=None, **kwargs):
        """ Call any function during pipeline execution

        Parameters
        ----------
        fn : a function, method or callable to call.
            Could be a named expression.

        save_to : a named expression or a sequence of named expressions
            A location where function output will be saved to.

        Notes
        -----
        As a function from any namespace (see :meth:`~.Pipeline.add_namespace`) can be called within a pipeline,
        `call` is convenient with lambdas::

            pipeline
                .call(lambda batch: (image.shape[1] for image in batch.images), B(), save_to=V('image_widths'))
        """
        return self._add_action(CALL_ID, *args, _args=dict(fn=fn, save_to=save_to), **kwargs)

    def _exec_call(self, batch, action):
        fn = self._eval_expr(action['fn'], batch=batch)
        if callable(fn):
            output = fn(*action['args'], **action['kwargs'])
        else:
            raise TypeError(f"Callable is expected, but got {type(fn)}")
        if action['save_to'] is not None:
            self._save_output(batch, None, output, action['save_to'])

    def _exec_from_ns(self, batch, action):
        res = action['method'](*action['args'], **action['kwargs'])
        if action['save_to'] is not None:
            self._save_output(batch, None, res, action['save_to'])

    @staticmethod
    def _get_action_method(batch, name):
        if hasattr(batch, name):
            attr = getattr(batch, name)
            if attr.__self__ == batch:
                # action decorator with arguments
                # attr is bounded to the batch
                action_method = attr
                action_attr = attr
            else:
                # action decorator wihout arguments
                action_method = attr
                action_attr = attr.__self__

            if callable(action_attr):
                if hasattr(action_attr, 'action'):
                    action_spec = getattr(action_attr, 'action')
                else:
                    raise ValueError(f"Method {name} is not marked with @action decorator")
            else:
                raise TypeError(f"{name} is not a method")
        else:
            raise AttributeError(f"Method '{name}' has not been found in the {type(batch).__name__} class")
        return action_method, action_spec

    def _exec_one_action(self, batch, action, iteration=None):
        if self._needs_exec(batch, action):
            repeat = self._eval_expr(action['repeat'], batch=batch) or 1
            for _ in range(repeat):

                # to save original args and kwargs with named expressions
                _action = action.copy()

                if action['name'] in ACTIONS:
                    action_method = getattr(self, ACTIONS[action['name']])
                    no_eval = None
                else:
                    action_method, action_spec = self._get_action_method(batch, action['name'])
                    no_eval = action_spec['no_eval']

                eval_time = time.time()
                if 'args' in action:
                    _action['args'] = self._eval_expr(action['args'], batch=batch)
                if 'kwargs' in action:
                    _action['kwargs'] = self._eval_expr(action['kwargs'], batch=batch, no_eval=no_eval)
                eval_time = time.time() - eval_time

                if action['name'] in ACTIONS:
                    action_method(batch, _action)
                else:
                    batch = action_method(*_action['args'], **_action['kwargs'])
                    if batch is not None:
                        batch.pipeline = self
                        batch.iteration = iteration
        # eval_time contains time of the last repeat only
        return batch, eval_time

    def _exec_nested_pipeline(self, batch, action):
        if self._needs_exec(batch, action):
            repeat = self._eval_expr(action['repeat'], batch=batch) or 1
            for _ in range(repeat):
                batch = self._exec_all_actions(batch, action['pipeline']._actions)  # pylint: disable=protected-access
        return batch

    def _exec_all_actions(self, batch, actions=None, iteration=None):
        join_batches = None
        actions = actions or self._actions

        batch.pipeline = self
        batch.iteration = iteration

        for action in actions:
            if self._profiler:
                self._profiler.enable()

            if action.get('#dont_run', False):
                pass
            elif action['name'] in [JOIN_ID, MERGE_ID]:
                join_batches = []
                for pipe in action['pipelines']:   # pylint: disable=not-an-iterable
                    if action['mode'] == 'i':
                        jbatch = pipe.create_batch(batch.index)
                    elif action['mode'] == 'n':
                        jbatch = pipe.next_batch()
                    join_batches.append(jbatch)
                join_batches = tuple(join_batches)

                if action['name'] == MERGE_ID:
                    if action['fn'] is None:
                        batch, _ = batch.merge([batch] + join_batches, components=action['components'])
                    else:
                        batch, _ = action['fn']([batch] + join_batches)
                    join_batches = None
            elif action['name'] == REBATCH_ID:
                pass
            elif action['name'] == PIPELINE_ID:
                batch = self._exec_nested_pipeline(batch, action)
            elif action['name'] == DISCARD_BATCH_ID:
                batch = None
            else:
                if join_batches is not None:
                    action['args'] = join_batches + action['args']
                    join_batches = None

                batch, action_time = self._exec_one_action(batch, action, iteration=iteration)

            if self._profiler:
                name = self.get_action_name(action, add_index=True)
                self._profiler.disable(batch.iteration, name, batch_id=id(batch), action_time=action_time)

        return batch

    def show_profile_info(self, **kwargs):
        return self._profiler.show_profile_info(**kwargs)

    @property
    def profile_info(self):
        return self._profiler.profile_info

    def _needs_exec(self, batch, action):
        if action['proba'] is None:
            return True
        proba = self._eval_expr(action['proba'], batch=batch)
        return self.random.binomial(1, proba) == 1

    def execute_for(self, batch, notifier=None, iteration=None, seed=None, new_loop=False):
        """ Run a pipeline for one batch

        Parameters
        ----------
        batch
            an input batch

        notifier
            a notifier instance

        iteration : int
            a pipeline iteration this batch is used at

        seed : SeedSequence
            a numpy SeedSequence to use when executing

        new_loop : bool
            whether to create a new :class:`async loop <asyncio.BaseEventLoop>`.

        Returns
        -------
        a batch - an output from the last action in the pipeline
        """
        if new_loop:
            asyncio.set_event_loop(asyncio.new_event_loop())
        if seed:
            self.random_seed = seed
        batch_res = self._exec_all_actions(batch, iteration=iteration)
        if notifier:
            notifier.update(pipeline=self, batch=batch_res)
        return batch_res

    def _eval_expr(self, expr, batch=None, no_eval=None):
        return eval_expr(expr, batch=batch, pipeline=self, no_eval=no_eval)

    def get_model_by_name(self, name, batch=None):
        """ Retrieve a model by its name """
        name = self._eval_expr(name, batch=batch)
        return self.models.get_model_by_name(name, batch=batch)

    def m(self, name, batch=None):
        """ A shorter alias for get_model_by_name() """
        return self.get_model_by_name(name, batch=batch)

    @property
    def model(self):
        """ An alias for a present model, if pipeline has only one model initialized. """
        n_models = len(self.models.models)
        if n_models == 1:
            return list(self.models.models.values())[0]
        raise TypeError('`model` property should be used only for pipelines with exactly 1 initialized model, '
                        f'but pipeline has {n_models} instead!')

    def init_model(self, name, model_class=None, mode='dynamic', config=None, source=None):
        """ Initialize a static or dynamic model by building or importing it

        Parameters
        ----------
        name : str
            a name for the model (to refer to it later when training or infering).

        model_class : class or named expression
            a model class (might also be specified in the config).

        mode : {'static', 'dynamic'}
            model creation mode:
            - static - the model is created right now, during the pipeline definition
            - dynamic - the model is created at the first iteration when the pipeline is run (default)

        config : dict or Config
            model configurations parameters, where each key and value could be named expressions.

        source
            a model or a pipeline to import from

        Examples
        --------
        Build a model::

            pipeline.init_model('my-model', MyModel, 'static')

        Import a model::

            pipeline.init_model('my-model', source=train_pipeline)

        Build a model with a config::

            pipeline
              .init_variable('images_shape', [256, 256])
              .init_model('my_model', MyModel, 'static', config={'input_shape': V('images_shape')})

            pipeline
              .init_variable('shape_name', 'images_shape')
              .init_model('my_model', C('model'), 'dynamic', config={V('shape_name'): B('images_shape')})

        """
        self.before.init_model(name, model_class, mode=mode, config=config, source=source)
        return self

    def import_model(self, name, source):
        """ Import a model from another pipeline

        Parameters
        ----------
        name : str
            a name with which the model is stored in this pipeline

        source
            a model or a pipeline to import from

        Examples
        --------
        Import a model instance to the pipeline::

            pipeline.import_model('my-model', custom_resnet_model)

        Import 'resnet' model from train_pipeline and store it in the pipeline under the name 'my-model'::

            pipeline.import_model('my-model', train_pipeline.m('resnet'))

        Import 'my-model' from train_pipeline and store it as 'my-model' in the pipeline::

            pipeline.import_model('my-model', train_pipeline)
        """
        return self._add_action(IMPORT_MODEL_ID, _args=dict(source=source, model_name=name))

    def _exec_import_model(self, batch, action):
        model_name = self._eval_expr(action['model_name'], batch=batch)
        source = self._eval_expr(action['source'], batch=batch)
        self.models.import_model(model_name, source)

    def train_model(self, name, *args, save_to=None, **kwargs):
        """ Train a model

        Parameters
        ----------
        name : str
            a model name

        save_to : a named expression or a sequence of named expressions.
            A location where the model output will be stored.

        Notes
        -----
        All other named parameters are treated as data mappings of any type
        which keys and values could be named expressions:

        - B('name') - a batch class attribute or component name
        - V('name') - a pipeline variable name
        - C('name') - a pipeline config option
        - F(name) - a callable
        - R('name') - a random value from a given distribution

        These expressions are substituted by their actual values.
        All other value will be used "as is".
        These parameters after substitution will be sent to `model.train(...)`.

        Examples
        --------
        >>> pipeline.train_model('resnet', x=B('images'), y_true=B('masks'))

        Would call a `resnet` model `train` method with `x` and `y_true` arguments:
        ``resnet.train(x=batch.images, y_true=batch.masks)``

        >>> pipeline
               .init_variable('tensor_name', 'x')
               .train_model('resnet', feed_dict={V('tensor_name'): B('images')})

        Would call a `resnet` model `train` method with a `feed_dict` argument:
        ``resnet.train(feed_dict={'x': batch.images})``
        """
        return self._add_action(TRAIN_MODEL_ID, *args, _args=dict(model_name=name, save_to=save_to), **kwargs)

    def predict_model(self, name, *args, save_to=None, **kwargs):
        """ Predict using a model

        Parameters
        ----------
        name : str - a model name

        save_to : a named expression or a sequence of named expressions.
            A location where the model output will be stored.

        Notes
        -----
        All other named parameters are treated as data mappings of any type
        which keys and values could be named expressions:

        - B('name') - a batch class attribute or component name
        - V('name') - a pipeline variable name
        - C('name') - a pipeline config option
        - F(name) - a callable
        - R('name') - a random value from a distribution 'name'

        These expressions are substituted by their actual values.
        All other value will be used "as is".
        These parameters after substitution will be sent to `model.predict(...)`.

        Examples
        --------
        >>> pipeline
                .predict_model('resnet', x=B('images'), y_true=B('labels'), save_to=B('predicted_labels'))

        Call a `resnet` model `predict` method with `x` and `y_true` arguments:
        ``predictions = resnet.predict(x=batch.images, y_true=batch.labels)``

        Predictions will be stored `batch.predicted_labels`.

        >>> pipeline
            .init_variable('inferred_masks', default=[])
            .predict_model('my_model',
                           B.images,
                           fetches='predictions',
                           save_to=V('inferred_masks'))

        Call a `my_model` model `train` method with images as positional and `fetches` as keyword arguments:
        ``predictions = my_model.train(B.images, fetches='predictions')``
        Predictions for each batch will be stored in a pipeline variable `inferred_masks`.
        """
        return self._add_action(PREDICT_MODEL_ID, *args, _args=dict(model_name=name, save_to=save_to), **kwargs)

    def _make_model_args(self, batch, action):
        args = self._eval_expr(action['args'], batch=batch)
        kwargs = self._eval_expr(action['kwargs'], batch=batch)
        return args, kwargs

    def _save_output(self, batch, model, output, locations):
        save_data_to(data=output, dst=locations, batch=batch, model=model)

    def _exec_train_model(self, batch, action):
        model = self.get_model_by_name(action['model_name'], batch=batch)
        args, kwargs = self._make_model_args(batch, action)
        output = model.train(*args, **kwargs)
        self._save_output(batch, model, output, action['save_to'])

    def _exec_predict_model(self, batch, action):
        model = self.get_model_by_name(action['model_name'], batch=batch)
        args, kwargs = self._make_model_args(batch, action)
        predictions = model.predict(*args, **kwargs)
        self._save_output(batch, model, predictions, action['save_to'])


    def load_model(self, name, model_class=None, mode='dynamic', *args, **kwargs):
        """ Load a model at each iteration

        Parameters
        ----------
        name : str
            a name for the model (to refer to it later when training or infering).

        model_class : class or named expression
            a model class (might also be specified in the config).

        mode : {'static', 'dynamic'}
            model creation mode:
            - static - the model is created right now, during the pipeline definition
            - dynamic - the model is created at the first iteration when the pipeline is run (default)

        args, kwargs
            model-specific parameters (like paths, formats, etc)
        """
        if mode == 'static':
            self.models.load_model(mode, name, model_class, *args, **kwargs)
            return self
        return self._add_action(LOAD_MODEL_ID, *args, _args=dict(mode=mode, model_class=model_class, model_name=name),
                                **kwargs)

    def load_model_once(self, mode, name=None, model_class=None, *args, **kwargs):
        """ Load a model once before the first iteration

        Parameters
        ----------
        name : str
            a name for the model (to refer to it later when training or infering).

        model_class : class or named expression
            a model class (might also be specified in the config).

        mode : {'static', 'dynamic'}
            model creation mode:
            - static - the model is created right now, during the pipeline definition
            - dynamic - the model is created at the first iteration when the pipeline is run (default)

        args, kwargs
            model-specific parameters (like paths, formats, etc)
        """
        self.before.load_model(mode, name, model_class, *args, **kwargs)
        return self

    def _exec_load_model(self, batch, action):
        mode = self._eval_expr(action['mode'], batch=batch)
        name = self._eval_expr(action['model_name'], batch=batch)
        model_class = self._eval_expr(action['model_class'], batch=batch)
        args, kwargs = self._make_model_args(batch, action)
        self.models.load_model(name, model_class, mode, *args, **kwargs)

    def load_model_now(self, name, model_class=None, mode='dynamic', *args, batch=None, **kwargs):
        """ Load a model immediately

        Parameters
        ----------
        name : str
            a name for the model (to refer to it later when training or infering).

        model_class : class or named expression
            a model class (might also be specified in the config).

        mode : {'static', 'dynamic'}
            model creation mode:
            - static - the model is created right now, during the pipeline definition
            - dynamic - the model is created at the first iteration when the pipeline is run (default)

        batch : Batch
            (optional) a batch which might be used to evaluate named expressions in other parameters

        args, kwargs
            model-specific parameters (like paths, formats, etc)
        """
        self._exec_load_model(batch, dict(mode=mode, model_name=name, model_class=model_class,
                                          args=args, kwargs=kwargs))

    def save_model(self, name, *args, **kwargs):
        """ Save a model at each iteration

        Parameters
        ----------
        name : str
            a model name

        args, kwargs
            model-specific parameters (like paths, formats, etc)
        """
        return self._add_action(SAVE_MODEL_ID, *args, _args=dict(model_name=name), **kwargs)

    def save_model_once(self, name, *args, **kwargs):
        """ Save a model after the last iteration

        Parameters
        ----------
        name : str
            a model name

        args, kwargs
            model-specific parameters (like paths, formats, etc)
        """
        self.after.save_model(name, *args, **kwargs)
        return self

    def _exec_save_model(self, batch, action):
        name = self._eval_expr(action['model_name'], batch=batch)
        args, kwargs = self._make_model_args(batch, action)
        self.models.save_model(name, *args, **kwargs)

    def save_model_now(self, name, *args, batch=None, **kwargs):
        """ Save a model immediately

        Parameters
        ----------
        name : str
            a model name

        batch : Batch
            (optional) a batch which might be used to evaluate named expressions in other parameters

        args, kwargs
            model-specific parameters (like paths, formats, etc)
        """
        self._exec_save_model(batch, dict(model_name=name, args=args, kwargs=kwargs))

    def gather_metrics(self, metrics_class, *args, save_to=None, **kwargs):
        """ Collect metrics for a model

        Parameters
        ----------
        metrics_class : class or str
            A class which calculates metrics (see :class:`~.Metrics`)

            If str:

            - 'class' for `:class:`~.ClassificationMetrics`)
            - 'segmentation' or 'mask' for `:class:`~.SegmentationMetricsByPixels`)
            - 'instance' for `:class:`~.SegmentationMetricsByInstances`)

        args
        kwargs
            Parameters for metrics calculation

        save_to : a named expression
            A location where metrics will be saved to.

        Notes
        -----
        For available metrics see :class:`metrics API <.metrics.Metrics>`.

        A mode can be passed to `save_to` expression:

        - 'w' saves metrics for the last batch only which is convenient for metrics evaluation during training.

        - 'u' is more suitable to calculate metrics during testing / validation.

        - 'a' collects the history of batch metrics.

        Examples
        --------

        ::

            pipeline = (dataset.test.p
                .init_variable('metrics')
                .init_variable('inferred_masks')
                .import_model('unet', train_pipeline)
                .predict_model('unet', fetches='predictions', feed_dict={'x': B('images')},
                               save_to=V('inferred_masks'))
                .gather_metrics('masks', targets=B('masks'), predictions=V('inferred_masks'),
                                fmt='proba', axis=-1, save_to=V('metrics', mode='u'))
                .run(BATCH_SIZE, notifier=True)
            )

            metrics = pipeline.get_variable('metrics')
            metrics.evaluate(['sensitivity', 'specificity'])
        """
        return self._add_action(GATHER_METRICS_ID, *args,
                                _args=dict(metrics_class=metrics_class, save_to=save_to),
                                **kwargs)

    def _exec_gather_metrics(self, batch, action):
        metrics_class = self._eval_expr(action['metrics_class'], batch=batch)
        if isinstance(metrics_class, str):
            available_metrics = [m for m in METRICS if metrics_class in m]
            if len(available_metrics) > 1:
                raise ValueError('Metrics name is ambiguous', metrics_class)
            if len(available_metrics) == 0:
                raise ValueError('Metrics not found', metrics_class)
            metrics_class = METRICS[available_metrics[0]]
        elif not isinstance(metrics_class, type):
            raise TypeError('Metrics can be a string or a class', metrics_class)

        metrics = metrics_class(*action['args'], **action['kwargs'])
        self._save_output(batch, None, metrics, action['save_to'])

    def join(self, *pipelines):
        """ Join one or several pipelines """
        return self._add_action(JOIN_ID, _args=dict(pipelines=pipelines, mode='i'))

    def merge(self, *pipelines, fn=None, components=None, batch_class=None):
        """ Merge pipelines """
        return self._add_action(MERGE_ID, _args=dict(pipelines=pipelines, mode='n', fn=fn,
                                                     components=components, batch_class=batch_class))

    def rebatch(self, batch_size, merge=None, components=None, batch_class=None):
        """ Set the output batch size """
        # pylint:disable=protected-access
        new_p = type(self)(self.dataset)
        return new_p._add_action(REBATCH_ID, _args=dict(batch_size=batch_size, pipeline=self, merge=merge,
                                                        components=components, batch_class=batch_class))

    def reset(self, *args, profile=False, seed=None):
        """ Clear all iteration metadata in order to start iterating from scratch

        Parameters
        ----------
        what : list of str, str or bool or None
            what to reset to start from scratch:

            - 'iter' - restart the batch iterator
            - 'variables' - re-initialize all pipeline variables
            - 'models' - reset all models

        profile : bool or {0, 1, 2} or 'detailed'
            whether to use profiler

        random
            a random state (see :func:`~.make_rng`).
            If not specified, RNG will be created with a random entropy.

        Examples
        --------
        ::

            pipeline.reset('iter')

            pipeline.reset('vars', 'models', profile=True)

            pipeline.reset(['iter', 'vars'], random=42)

        """
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            args = args[0]
        what = args

        if len(what) == 1:
            if what[0] is None or what[0] is False:
                what = []
            elif what[0] is True:
                # reset(True) is reset('iter')
                what = 'iter'
            elif what[0] == 'all':
                what = ['iter', 'variables', 'models']
        if isinstance(what, str):
            what = [what]

        if 'iter' in what:
            self.iter_params = Baseset.get_default_iter_params()
            self._batch_generator = None

        if 'vars' in what or 'variables' in what or self.variables_initialised:
            # initialise all variables before the very first run of this pipeline
            self._init_all_variables()
            self.variables_initialised = True

        if 'models' in what:
            self.models.reset()

        self.random_seed = seed

        if profile == 2 or isinstance(profile, str) and 'detailed'.startswith(profile):
            self._profiler = PipelineProfiler(detailed=True)
        elif profile == 1 or profile is True:
            self._profiler = PipelineProfiler(detailed=False)
        else: # 0, False, None
            self._profiler = None


    def gen_rebatch(self, *args, **kwargs):
        """ Generate batches for rebatch operation """
        _action = self._actions[0]

        if _action['pipeline'].dataset is None:
            pipeline = _action['pipeline'] << self._dataset
        else:
            pipeline = self.from_pipeline(_action['pipeline'])

        self._rest_batch = None
        while True:
            if self._rest_batch is None:
                cur_len = 0
                batches = []
            else:
                cur_len = len(self._rest_batch)
                batches = [self._rest_batch]
                self._rest_batch = None

            while cur_len < _action['batch_size']:
                try:
                    new_batch = pipeline.next_batch(*args, **kwargs)
                except (StopIteration, StopPipeline):
                    break
                else:
                    batches.append(new_batch)
                    cur_len += len(new_batch)

            if len(batches) == 0:
                break

            if _action['merge'] is None:
                batch, self._rest_batch = batches[0].merge(batches, batch_size=_action['batch_size'],
                                                           components=_action['components'],
                                                           batch_class=_action['batch_class'])
            else:
                batch, self._rest_batch = _action['merge'](batches, batch_size=_action['batch_size'],
                                                           components=_action['components'],
                                                           batch_class=_action['batch_class'])
            yield batch


    def _eval_run_args(self, args, kwargs):
        if len(args) == 0 and len(kwargs) == 0:
            if self._lazy_run is None:
                raise RuntimeError("a pipeline run without arguments requires a lazy run at the end of the pipeline")
            args, kwargs = self._lazy_run

        if self._lazy_run:
            # if lazy args were set, then use them now
            _args, _kwargs = self._lazy_run
            # but update args (usually batch_size)
            args = _args if len(args) == 0 else args
            kwargs = {**_kwargs, **kwargs}

        self._dataset = self._eval_expr(self.dataset)
        args_value = self._eval_expr(args)
        kwargs_value = self._eval_expr(kwargs)

        return args_value, kwargs_value

    def gen_batch(self, *args, **kwargs):
        """ Generate batches

        Parameters
        ----------
        batch_size : int
            desired number of items in the batch (the actual batch could contain fewer items)

        shuffle : bool or int
            specifies the randomization and the order of items (default=False):

            - `False` - items go sequentially, one after another as they appear in the index;
              a random number generator is created with a random entropy

            - `True` - items are shuffled randomly before each epoch;
              a random number generator is created with a random entropy

            - int - a seed number for a random shuffle;
              a random number generator is created with the given seed.

        n_iters : int
            Number of iterations to make (only one of `n_iters` and `n_epochs` should be specified).

        n_epochs : int
            Number of epochs required (only one of `n_iters` and `n_epochs` should be specified).

        drop_last : bool
            if `True`, drops the last batch (in each epoch) if it contains fewer than `batch_size` items.

            If `False`, than the last batch in each epoch could contain repeating indices (which might be a problem)
            and the very last batch could contain fewer than `batch_size` items.

            See :meth:`~.DatasetIndex.gen_batch` for details.

        notifier : str, dict, or instance of `.Notifier`
            Configuration of displayed progress bar, if any.
            If str or dict, then parameters of `.Notifier` initialization.
            For more details about notifying capabilities, refer to `.Notifier` documentation.

        prefetch : int
            a number of batches to process in advance (default=0)

        target : 'threads' or 'mpc'
            batch parallelization engine used for prefetching (default='threads').
            'mpc' rarely works well due to complicated and slow python's inter-process communications.
            Don't use pipeline variables and models in mpc-mode as each batch is being processed in
            a separate copy of the pipeline.

        reset : list of str, str or bool
            what to reset to start from scratch:

            - 'iter' - restart the batch iterator
            - 'variables' - re-initialize all pipeline variables
            - 'models' - reset all models

        ignore_exceptions : bool
            whether to continue the pipeline when an exception for any batch is caught (default=True).
            When exceptions are not ignored while prefetching, the pipeline is stopped when the first one is caught,
            however, all prefeteched batches will still be processed in the background.

        profile
            whether to gather execution statistics.
            0 or False - do not gather
            1 or True - gather action times
            2 or 'detailed' - gather full profiling with cProfile.

        Yields
        ------
        an instance of the batch class returned by the last action

        Examples
        --------

        ::

            for batch in pipeline.gen_batch(C('batch_size'), shuffle=True, n_epochs=2, drop_last=True):
                # do whatever you want
        """
        args_value, kwargs_value = self._eval_run_args(args, kwargs)

        rebatch = len(self._actions) > 0 and self._actions[0]['name'] == REBATCH_ID
        if rebatch:
            _, self._actions[0] = self._eval_run_args([], self._actions[0])

        return PipelineExecutor(self).gen_batch(*args_value, dataset=self._dataset, rebatch=rebatch, **kwargs_value)


    def create_batch(self, batch_index, *args, **kwargs):
        """ Create a new batch by given indices and execute all lazy actions """
        batch = self._dataset.create_batch(batch_index, *args, **kwargs)
        batch_res = self.execute_for(batch)
        return batch_res

    def next_batch(self, *args, n_epochs=None, **kwargs):
        """ Get the next batch and execute all lazy actions

        Notes
        -----
        `n_epochs` is None by default to allow for infinite batch generation.

        See also
        --------
        :meth:`~.Pipeline.gen_batch`
        """
        if len(args) == 0 and len(kwargs) == 0 and self._lazy_run is not None:
            args, kwargs = self._lazy_run
        else:
            kwargs['n_epochs'] = n_epochs

        if self._batch_generator is None:
            self._batch_generator = self.gen_batch(*args, reset=None, **kwargs)
        batch = next(self._batch_generator)

        return batch

    def run(self, *args, **kwargs):
        """ Execute all lazy actions for each batch in the dataset

        See also
        --------
        :meth:`~.Pipeline.gen_batch`
        """
        if kwargs.pop('lazy', False):
            # if lazy is passed, then store params for later execution
            self._lazy_run = args, kwargs
            return self

        if len(args) == 0 and len(kwargs) == 0 and self._lazy_run is not None:
            args, kwargs = self._lazy_run

        args_value, kwargs_value = self._eval_run_args(args, kwargs)

        if kwargs_value.get('n_epochs', None) is None and kwargs_value.get('n_iters', None) is None:
            warnings.warn('Batch generation will never stop as ' \
                          'n_epochs=None and n_iters=None', RuntimeWarning)

        return PipelineExecutor(self).run(*args_value, **kwargs_value)

    def run_now(self, *args, **kwargs):
        """ Execute pipeline immediately """
        return self.run(*args, **kwargs, lazy=False)

    def run_later(self, *args, **kwargs):
        """ Define params to execute pipeline later """
        return self.run(*args, **kwargs, lazy=True)
