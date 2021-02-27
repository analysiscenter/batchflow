""" Contains pipeline class """
# pylint:disable=undefined-variable
import sys
import time
from functools import partial
import traceback
import threading
import concurrent.futures as cf
import asyncio
import logging
import warnings
from cProfile import Profile
from pstats import Stats
import queue as q
import numpy as np
try:
    import pandas as pd
except ImportError:
    from . import _fake as pd

from .base import Baseset
from .config import Config
from .batch import Batch
from .decorators import deprecated
from .exceptions import SkipBatchException, EmptyBatchSequence, StopPipeline
from .named_expr import NamedExpression, V, eval_expr
from .once_pipeline import OncePipeline
from .model_dir import ModelDirectory
from .variables import VariableDirectory
from .models.metrics import (ClassificationMetrics, SegmentationMetricsByPixels,
                             SegmentationMetricsByInstances, RegressionMetrics, Loss)

from ._const import *       # pylint:disable=wildcard-import
from .utils import save_data_to
from .notifier import Notifier


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
    def __init__(self, dataset=None, config=None, pipeline=None, actions=None, proba=None, repeat=None):
        # pylint: disable=protected-access

        if pipeline is None:
            self.dataset = dataset
            self.config = config or {}
            self._actions = actions or []
            self._lazy_run = None
            self.models = ModelDirectory()
            self.variables = VariableDirectory()
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
            self.models = pipeline.models.copy()
            self._namespaces = pipeline._namespaces
            self.before = pipeline.before.copy()
            self.before.pipeline = self
            self.after = pipeline.after.copy()
            self.after.pipeline = self

        self._dataset = None
        self.config = Config(self.config)
        self._stop_flag = False
        self._executor = None
        self._service_executor = None
        self._prefetch_count = None
        self._prefetch_queue = None
        self._batch_queue = None
        self._batch_generator = None
        self._rest_batch = None
        self._iter_params = None
        self._not_init_vars = True

        self.notifier = None
        self._profile = None
        self._profiler = None
        self.profile_info = None
        self.elapsed_time = 0.0
        self._profile_info_lock = threading.Lock()

    def __enter__(self):
        """ Create a context and return an empty pipeline non-bound to any dataset """
        return type(self)()

    def __exit__(self, exc_type, exc_value, trback):
        pass

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
        return name if add_index is False else '{} #{}'.format(name, idx)

    def add_namespace(self, *namespaces):
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
        return common_namespaces + self._namespaces

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
            raise AttributeError('Unknown magic method: %s' % name)
        if self._is_batch_method(name):
            return partial(self._add_action, name)
        if self.is_method_from_ns(name):
            return partial(self._add_action, CALL_FROM_NS_ID, _name=name)
        raise AttributeError("%s not found in class %s" % (name, self.__class__.__name__))

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

    def assign_variable(self, name, value, **kwargs):
        """ Assign a value to a variable """
        _ = kwargs
        if not self.has_variable(name):
            logging.warning("Pipeline variable '%s' has not been initialized", name)
            self.init_variable(name)
        self.variables.set(name, value)

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
                .call(lambda batch: [image.shape[1] for image in batch.images], save_to=V('image_widths'))
        """
        return self._add_action(CALL_ID, *args, _args=dict(fn=fn, save_to=save_to), **kwargs)

    def _exec_call(self, batch, action):
        fn = self._eval_expr(action['fn'], batch)
        if callable(fn):
            output = fn(batch, *action['args'], **action['kwargs'])
        else:
            raise TypeError("Callable is expected, but got {}".format(type(fn)))
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
                    raise ValueError("Method %s is not marked with @action decorator" % name)
            else:
                raise TypeError("%s is not a method" % name)
        else:
            raise AttributeError("Method '%s' has not been found in the %s class" % (name, type(batch).__name__))
        return action_method, action_spec

    def _exec_one_action(self, batch, action, args, kwargs):
        if self._needs_exec(batch, action):
            repeat = self._eval_expr(action['repeat'], batch=batch) or 1
            for _ in range(repeat):
                batch.pipeline = self
                action_method, _ = self._get_action_method(batch, action['name'])
                batch = action_method(*args, **kwargs)
                batch.pipeline = self
        return batch

    def _exec_nested_pipeline(self, batch, action):
        if self._needs_exec(batch, action):
            repeat = self._eval_expr(action['repeat'], batch=batch) or 1
            for _ in range(repeat):
                batch = self._exec_all_actions(batch, action['pipeline']._actions)  # pylint: disable=protected-access
        return batch

    def _add_profile_info(self, batch, action, exec_time, **kwargs):
        name = self.get_action_name(action, add_index=True)
        iter_no = self._iter_params['_n_iters']

        start_time = time.time()
        stats = Stats(self._profiler)
        self._profiler.clear()

        indices, values = [], []
        for key, value in stats.stats.items():
            for k, v in value[4].items():
                # action name, method_name, file_name, line_no, callee
                indices.append((name, '{}::{}::{}::{}'.format(key[2], *k)))
                row_dict = {
                    'iter': iter_no, 'total_time': exec_time, 'pipeline_time': stats.total_tt, # base stats
                    'ncalls': v[0], 'tottime': v[2], 'cumtime': v[3], # detailed stats
                    'batch_id': id(batch), **kwargs
                }
                values.append(row_dict)


        multiindex = pd.MultiIndex.from_tuples(indices, names=['action', 'id'])
        df = pd.DataFrame(values, index=multiindex,
                          columns=['iter', 'total_time', 'pipeline_time',
                                   'ncalls', 'tottime', 'cumtime',
                                   'batch_id', *list(kwargs.keys())])
        df['total_time'] += time.time() - start_time

        if self.profile_info is None:
            with self._profile_info_lock:
                self.profile_info = df
        else:
            with self._profile_info_lock:
                self.profile_info = self.profile_info.append(df)

    def show_profile_info(self, per_iter=False, detailed=False,
                          groupby=None, columns=None, sortby=None, limit=10):
        """ Show stored profiling information with varying levels of details.

        Parameters
        ----------
        per_iter : bool
            Whether to make an aggregation over iters or not.
        detailed : bool
            Whether to use information from :class:`cProfiler` or not.
        groupby : str or sequence of str
            Used only when `per_iter` is True, directly passed to pandas.
        columns : sequence of str
            Columns to show in resultining dataframe.
        sortby : str or tuple of str
            Column id to sort on. Note that if data is aggregated over iters (`per_iter` is False),
            then it must be a full identificator of a column.
        limit : int
            Limits the length of resulting dataframe.
        parse : bool
            Allows to re-create underlying dataframe from scratches.
        """
        if per_iter is False and detailed is False:
            columns = columns or ['total_time', 'pipeline_time']
            sortby = sortby or ('total_time', 'sum')
            aggs = {key: ['sum', 'mean', 'max'] for key in columns}
            result = (self.profile_info.groupby(['action', 'iter'])[columns].mean().groupby('action').agg(aggs)
                      .sort_values(sortby, ascending=False))

        elif per_iter is False and detailed is True:
            columns = columns or ['ncalls', 'tottime', 'cumtime']
            sortby = sortby or ('tottime', 'sum')
            aggs = {key: ['sum', 'mean', 'max'] for key in columns}
            result = (self.profile_info.reset_index().groupby(['action', 'id']).agg(aggs)
                      .sort_values(['action', sortby], ascending=[True, False])
                      .groupby(level=0).apply(lambda df: df[:limit]).droplevel(0))

        elif per_iter is True and detailed is False:
            groupby = groupby or ['iter', 'action']
            columns = columns or ['action', 'total_time', 'pipeline_time', 'batch_id']
            sortby = sortby or 'total_time'
            result = (self.profile_info.reset_index().groupby(groupby)[columns].mean()
                      .sort_values(['iter', sortby], ascending=[True, False]))

        elif per_iter is True and detailed is True:
            groupby = groupby or ['iter', 'action', 'id']
            columns = columns or ['ncalls', 'tottime', 'cumtime']
            sortby = sortby or 'tottime'
            result = (self.profile_info.reset_index().set_index(groupby)[columns]
                      .sort_values(['iter', 'action', sortby], ascending=[True, True, False])
                      .groupby(level=[0, 1]).apply(lambda df: df[:limit]).droplevel([0, 1]))
        return result


    def _exec_all_actions(self, batch, actions=None):
        join_batches = None
        actions = actions or self._actions

        for action in actions:
            if self._profile:
                start_time = time.time()
                self._profiler.enable()

            _action = action.copy()
            if 'args' in action:
                _action['args'] = self._eval_expr(action['args'], batch=batch)
            if 'kwargs' in action:
                _action['kwargs'] = self._eval_expr(action['kwargs'], batch=batch)

            if self._profile:
                eval_expr_time = time.time() - start_time

            if _action.get('#dont_run', False):
                pass
            elif _action['name'] in [JOIN_ID, MERGE_ID]:
                join_batches = []
                for pipe in _action['pipelines']:   # pylint: disable=not-an-iterable
                    if _action['mode'] == 'i':
                        jbatch = pipe.create_batch(batch.index)
                    elif _action['mode'] == 'n':
                        jbatch = pipe.next_batch()
                    join_batches.append(jbatch)

                if _action['name'] == MERGE_ID:
                    if _action['fn'] is None:
                        batch, _ = batch.merge([batch] + join_batches, components=_action['components'])
                    else:
                        batch, _ = _action['fn']([batch] + join_batches)
                    join_batches = None
            elif _action['name'] == REBATCH_ID:
                pass
            elif _action['name'] == PIPELINE_ID:
                batch = self._exec_nested_pipeline(batch, _action)
            elif _action['name'] in ACTIONS:
                action_fn = getattr(self, ACTIONS[_action['name']])
                action_fn(batch, _action)
            else:
                if join_batches is None:
                    _action_args = _action['args']
                else:
                    _action_args = tuple([tuple(join_batches), *_action['args']])
                    join_batches = None

                batch = self._exec_one_action(batch, _action, _action_args, _action['kwargs'])

            if self._profile:
                self._profiler.disable()
                exec_time = time.time() - start_time
                self._add_profile_info(batch, action, start_time=start_time, exec_time=exec_time,
                                       eval_expr_time=eval_expr_time)


        return batch

    def _needs_exec(self, batch, action):
        if action['proba'] is None:
            return True
        proba = self._eval_expr(action['proba'], batch=batch)
        return np.random.binomial(1, proba) == 1

    def execute_for(self, batch, new_loop=False):
        """ Run a pipeline for one batch

        Parameters
        ----------
        batch
            an input batch
        new_loop : bool
            whether to create a new :class:`async loop <asyncio.BaseEventLoop>`.

        Returns
        -------
        a batch - an output from the last action in the pipeline
        """
        if new_loop:
            asyncio.set_event_loop(asyncio.new_event_loop())
        batch.pipeline = self
        batch_res = self._exec_all_actions(batch)
        batch_res.pipeline = self
        return batch_res

    def _eval_expr(self, expr, batch=None):
        return eval_expr(expr, batch=batch, pipeline=self)

    def get_model_by_name(self, name, batch=None):
        """ Retrieve a model by its name """
        name = self._eval_expr(name, batch=batch)
        return self.models.get_model_by_name(name, batch=batch)

    def m(self, name, batch=None):
        """ A shorter alias for get_model_by_name() """
        return self.get_model_by_name(name, batch=batch)

    def init_model(self, mode, name=None, model_class=None, config=None):
        """ Initialize a static or dynamic model

        Parameters
        ----------
        mode : {'static', 'dynamic'}
        name : str
            (optional) a name for the model. Default - a model class name.
        model_class : class or named expression
            (optional) a model class (if not specified in the config).
        config : dict or Config
            (optional) model configurations parameters, where each key and value could be named expressions.

        Examples
        --------
        >>> pipeline.init_model('static', MyModel)

        >>> pipeline
              .init_variable('images_shape', [256, 256])
              .init_model('static', 'my_model', MyModel, config={'input_shape': V('images_shape')})

        >>> pipeline
              .init_variable('shape_name', 'images_shape')
              .init_model('dynamic', C('model'), config={V('shape_name)': B('images_shape')})

        >>> pipeline
              .init_model('dynamic', MyModel, config={'input_shape': C(lambda batch: batch.images.shape[1:])})
        """
        self.before.init_model(mode, name, model_class, config=config)
        return self

    def import_model(self, model, pipeline=None, name=None):
        """ Import a model from another pipeline

        Parameters
        ----------
        model : str or model
            a name of the model to import or a model itself
        pipeline : Pipeline
            a pipeline that holds a model
        name : str
            a name with which the model is stored in this pipeline
        """
        return self._add_action(IMPORT_MODEL_ID, _args=dict(source=model, pipeline=pipeline, model_name=name))

    def _exec_import_model(self, batch, action):
        model_name = self._eval_expr(action['model_name'], batch=batch)
        source = self._eval_expr(action['source'], batch=batch)
        pipeline = self._eval_expr(action['pipeline'], batch=batch)
        self.models.import_model(source, pipeline, model_name)

    def train_model(self, name, *args, make_data=None, save_to=None, **kwargs):
        """ Train a model

        Parameters
        ----------
        name : str
            a model name

        make_data : a callable or a named expression
            a function or method to transform batch data to train parameters.
            Should return dict - kwargs for `model.train(...)`.

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

        >>> pipeline.train_model('resnet', MyBatch.make_resnet_data)

        Equivalent to::

            train_data = batch.make_resnet_data(resnet_model)
            resnet_model.train(**train_data)
        """
        return self._add_action(TRAIN_MODEL_ID, *args,
                                _args=dict(model_name=name, make_data=make_data, save_to=save_to),
                                **kwargs)

    def predict_model(self, name, *args, make_data=None, save_to=None, **kwargs):
        """ Predict using a model

        Parameters
        ----------
        name : str - a model name

        make_data : a callable or a named expression
            a function or method to transform batch data to prediction parameters.
            Should return dict - kwargs for `model.predict(...)`.

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
            .predict_model('tf_unet', fetches='predictions', feed_dict={'x': B('images')},
                           save_to=V('inferred_masks'))

        Call a `tf_unet` model `train` method with `fetches` and `feed_dict` arguments:
        ``predictions = tf_unet.train(fetches='predictions', feed_dict={'x': batch.images})``
        Predictions for each batch will be stored in a pipeline variable `inferred_masks`.

        >>> pipeline.predict_model('deepnet', MyBatch.make_deepnet_data)

        Equivalent to::

            predict_data = batch.make_deepnet_data(model=deepnet_model)
            deepnet_model.predict(**predict_data)
        """
        return self._add_action(PREDICT_MODEL_ID, *args,
                                _args=dict(model_name=name, make_data=make_data, save_to=save_to),
                                **kwargs)

    def _make_model_args(self, batch, action, model):
        make_data = action.get('make_data') or  {}
        args = action['args']
        kwargs = dict()

        if callable(make_data):
            kwargs = make_data(batch=batch, model=model)
        else:
            kwargs = self._eval_expr(make_data, batch=batch)
        if not isinstance(kwargs, dict):
            raise TypeError("make_data should return a dict with kwargs", make_data)

        kwargs = {**action['kwargs'], **kwargs}

        kwargs = self._eval_expr(kwargs, batch=batch)

        return args, kwargs

    def _save_output(self, batch, model, output, locations):
        save_data_to(data=output, dst=locations, batch=batch, model=model)

    def _exec_train_model(self, batch, action):
        model = self.get_model_by_name(action['model_name'], batch=batch)
        args, kwargs = self._make_model_args(batch, action, model)
        output = model.train(*args, **kwargs)
        self._save_output(batch, model, output, action['save_to'])

    def _exec_predict_model(self, batch, action):
        model = self.get_model_by_name(action['model_name'], batch=batch)
        args, kwargs = self._make_model_args(batch, action, model)
        predictions = model.predict(*args, **kwargs)
        self._save_output(batch, model, predictions, action['save_to'])

    def load_model(self, mode, name=None, model_class=None, *args, **kwargs):
        """ Load a model at each iteration

        Parameters
        ----------
        mode : str
            'static' or 'dynamic'

        name : str
            (optional) a model name

        model_class : class or named expression
            (optional) a model class to instantiate a loaded model instance.

        args, kwargs
            model-specific parameters (like paths, formats, etc)
        """
        if mode == 'static':
            self.models.load_model(mode, name, model_class, *args, **kwargs)
            return self
        return self._add_action(LOAD_MODEL_ID, *args,
                                _args=dict(mode=mode, model_class=model_class, model_name=name),
                                **kwargs)

    def load_model_once(self, mode, name=None, model_class=None, *args, **kwargs):
        """ Load a model once before the first iteration

        Parameters
        ----------
        mode : str
            'static' or 'dynamic'

        name : str
            (optional) a model name

        model_class : class or named expression
            (optional) a model class to instantiate a loaded model instance.

        args, kwargs
            model-specific parameters (like paths, formats, etc)
        """
        self.before.load_model(mode, name, model_class, *args, **kwargs)
        return self

    def _exec_load_model(self, batch, action):
        mode = self._eval_expr(action['mode'], batch=batch)
        name = self._eval_expr(action['model_name'], batch=batch)
        model_class = self._eval_expr(action['model_class'], batch=batch)
        args, kwargs = self._make_model_args(batch, action, None)
        self.models.load_model(mode, name, model_class, *args, **kwargs)

    def load_model_now(self, mode, name=None, model_class=None, *args, batch=None, **kwargs):
        """ Load a model immediately

        Parameters
        ----------
        mode : str
            'static' or 'dynamic'

        name : str
            (optional) a model name

        model_class : class or named expression
            (optional) a model class to instantiate a loaded model instance.

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
        model = self.get_model_by_name(name)
        args, kwargs = self._make_model_args(batch, action, model)
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
        metrics_class = self._eval_expr(action['metrics_class'], batch)
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

    def rebatch(self, batch_size, fn=None, components=None, batch_class=None):
        """ Set the output batch size """
        # pylint:disable=protected-access
        new_p = type(self)(self.dataset)
        return new_p._add_action(REBATCH_ID, _args=dict(batch_size=batch_size, pipeline=self, fn=fn,
                                                        components=components, batch_class=batch_class))

    def _put_batches_into_queue(self, gen_batch):
        while not self._stop_flag:
            self._prefetch_count.put(1, block=True)
            try:
                batch = next(gen_batch)
            except StopIteration:
                break
            else:
                future = self._executor.submit(self.execute_for, batch, new_loop=True)
                self._prefetch_queue.put(future, block=True)
        self._prefetch_queue.put(None, block=True)

    def _run_batches_from_queue(self, notifier):
        skip_batch = False
        while not self._stop_flag:
            future = self._prefetch_queue.get(block=True)
            if future is None:
                self._prefetch_queue.task_done()
                self._batch_queue.put(None)
                break

            try:
                batch = future.result()
                notifier.update(pipeline=self, batch=batch)
            except SkipBatchException:
                skip_batch = True
            except StopPipeline:
                self._prefetch_queue.task_done()
                self._batch_queue.put(None)
                break
            except Exception:   # pylint: disable=broad-except
                exc = future.exception()
                print("Exception in a thread:", exc)
                traceback.print_tb(exc.__traceback__)
            finally:
                if not skip_batch:
                    self._batch_queue.put(batch, block=True)
                    skip_batch = False
                self._prefetch_queue.task_done()

    def _clear_queue(self, queue):
        if queue is not None:
            while not queue.empty():
                queue.get(block=True)
                queue.task_done()

    def _stop_executor(self, executor):
        if executor is not None:
            executor.shutdown()

    def reset(self, *args):
        """ Clear all iteration metadata in order to start iterating from scratch

        Parameters
        ----------
        what : list of str, str or bool or None
            what to reset to start from scratch:

            - 'iter' - restart the batch iterator
            - 'variables' - re-initialize all pipeline variables
            - 'models' - reset all models

        Examples
        --------
        ::
            pipeline.reset('iter')

            pipeline.reset('vars', 'models')

            pipeline.reset(['iter', 'vars'])

        """
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            args = args[0]
        what = args

        if len(what) == 1:
            if what[0] is None or what[0] is False:
                what = []
            elif what[0] is True:
                what = 'iter'
            elif what[0] == 'all':
                what = ['iter', 'variables', 'models']
        if isinstance(what, str):
            what = [what]

        if 'iter' in what:
            self._stop_flag = True

            self._clear_queue(self._prefetch_queue)
            self._clear_queue(self._batch_queue)
            self._clear_queue(self._prefetch_count)

            self._stop_executor(self._executor)
            self._stop_executor(self._service_executor)

            self._executor = None
            self._service_executor = None
            self._prefetch_count = None
            self._prefetch_queue = None
            self._batch_queue = None
            self._rest_batch = None
            self._batch_generator = None
            self._iter_params = Baseset.get_default_iter_params()

        if 'vars' in what or 'variables' in what:
            self._init_all_variables()

        if 'models' in what:
            self.models.reset()


    def gen_rebatch(self, *args, **kwargs):
        """ Generate batches for rebatch operation """
        _action = self._actions[0]

        if _action['pipeline'].dataset is None:
            pipeline = _action['pipeline'] << self._dataset
        else:
            pipeline = self.from_pipeline(_action['pipeline'])

        kwargs.setdefault('iter_params', None)

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

            if _action['fn'] is None:
                batch, self._rest_batch = batches[0].merge(batches, batch_size=_action['batch_size'],
                                                           components=_action['components'],
                                                           batch_class=_action['batch_class'])
            else:
                batch, self._rest_batch = _action['fn'](batches, batch_size=_action['batch_size'],
                                                        components=_action['components'],
                                                        batch_class=_action['batch_class'])
            yield batch


    def gen_batch(self, *args, iter_params=None, reset='iter', profile=False, **kwargs):
        """ Generate batches

        Parameters
        ----------
        batch_size : int
            desired number of items in the batch (the actual batch could contain fewer items)

        shuffle : bool, int, class:`numpy.random.RandomState` or callable
            specifies the order of items, could be:

            - bool - if `False`, items go sequentionally, one after another as they appear in the index.
                if `True`, items are shuffled randomly before each epoch.

            - int - a seed number for a random shuffle.

            - :class:`numpy.random.RandomState` instance.

            - callable - a function which takes an array of item indices in the initial order
                (as they appear in the index) and returns the order of items.

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

        reset : list of str, str or bool
            what to reset to start from scratch:

            - 'iter' - restart the batch iterator
            - 'variables' - re-initialize all pipeline variables
            - 'models' - reset all models

        Yields
        ------
        an instance of the batch class returned by the last action

        Examples
        --------

        ::

            for batch in pipeline.gen_batch(C('batch_size'), shuffle=True, n_epochs=2, drop_last=True):
                # do whatever you want
        """
        if len(args) == 0 and len(kwargs) == 0:
            if self._lazy_run is None:
                raise RuntimeError("gen_batch without arguments requires a lazy run at the end of the pipeline")
            args, kwargs = self._lazy_run

        self._dataset = self._eval_expr(self.dataset)
        args_value = self._eval_expr(args)
        kwargs_value = self._eval_expr(kwargs)
        self.reset(reset)
        self._iter_params = iter_params or self._iter_params or Baseset.get_default_iter_params()
        self._profile = profile
        if profile:
            self._profiler = Profile()

        return self._gen_batch(*args_value, iter_params=self._iter_params, **kwargs_value)


    def _gen_batch(self, *args, **kwargs):
        """ Generate batches """
        start_time = time.time()
        target = kwargs.pop('target', 'threads')
        prefetch = kwargs.pop('prefetch', 0)
        on_iter = kwargs.pop('on_iter', None)
        if 'bar' in kwargs:
            warnings.warn('`bar` argument is deprecated and renamed to `notifier`', DeprecationWarning, stacklevel=2)
        notifier = kwargs.pop('notifier', kwargs.pop('bar', None))
        total = kwargs.pop('total', None)

        if len(self._actions) > 0 and self._actions[0]['name'] == REBATCH_ID:
            batch_generator = self.gen_rebatch(*args, **kwargs, prefetch=prefetch)
            prefetch = 0
        else:
            batch_generator = self._dataset.gen_batch(*args, **kwargs)

        if self._not_init_vars:
            self._init_all_variables()
            self._not_init_vars = False

        batch_size = args[0] if len(args) != 0 else kwargs.get('batch_size')
        n_iters = kwargs.get('n_iters')
        n_epochs = kwargs.get('n_epochs')
        drop_last = kwargs.get('drop_last')

        if not isinstance(notifier, Notifier):
            notifier = Notifier(**(notifier if isinstance(notifier, dict) else {'bar': notifier}),
                                total=total, batch_size=batch_size, n_iters=n_iters, n_epochs=n_epochs,
                                drop_last=drop_last, length=len(self._dataset.index))
        elif notifier.total is None:
            notifier.update_total(total=None, batch_size=batch_size, n_iters=n_iters, n_epochs=n_epochs,
                                  drop_last=drop_last, length=len(self._dataset.index))

        if self.before:
            self.before.run()

        if prefetch > 0:
            # pool cannot have more than 63 workers
            prefetch = min(prefetch, 62)

            if target in ['threads', 't']:
                self._executor = cf.ThreadPoolExecutor(max_workers=prefetch + 1)
            elif target in ['mpc', 'm']:
                self._executor = cf.ProcessPoolExecutor(max_workers=prefetch + 1)
            else:
                raise ValueError("target should be one of ['threads', 'mpc']")

            self._stop_flag = False
            self._prefetch_count = q.Queue(maxsize=prefetch + 1)
            self._prefetch_queue = q.Queue(maxsize=prefetch)
            self._batch_queue = q.Queue(maxsize=1)
            self._service_executor = cf.ThreadPoolExecutor(max_workers=2)
            self._service_executor.submit(self._put_batches_into_queue, batch_generator)
            self._service_executor.submit(self._run_batches_from_queue, notifier)

            while not self._stop_flag:
                batch_res = self._batch_queue.get(block=True)
                self._batch_queue.task_done()
                if batch_res is not None:
                    yield batch_res
                    self._prefetch_count.get(block=True)
                    self._prefetch_count.task_done()
                    if callable(on_iter):
                        on_iter(batch_res)
                else:
                    self._stop_flag = True
        else:
            is_empty = True
            for batch in batch_generator:
                try:
                    batch_res = self.execute_for(batch)
                    notifier.update(pipeline=self, batch=batch_res)
                except SkipBatchException:
                    pass
                except StopPipeline:
                    break
                else:
                    is_empty = False
                    yield batch_res
                    if callable(on_iter):
                        on_iter(batch_res)
            if is_empty:
                warnings.warn("Batch generator is empty. Use pipeline.reset('iter') to restart iteration.",
                              EmptyBatchSequence, stacklevel=3)

        notifier.close()
        self.notifier = notifier

        if self.after:
            self.after.run()
        self.elapsed_time += time.time() - start_time


    def create_batch(self, batch_index, *args, **kwargs):
        """ Create a new batch by given indices and execute all lazy actions """
        batch = self._dataset.create_batch(batch_index, *args, **kwargs)
        batch_res = self.execute_for(batch)
        return batch_res

    def next_batch(self, *args, **kwargs):
        """ Get the next batch and execute all lazy actions

        See also
        --------
        :meth:`~.Pipeline.gen_batch`
        """
        start_time = time.time()
        if len(args) == 0 and len(kwargs) == 0:
            if self._lazy_run is None:
                raise RuntimeError("next_batch without arguments requires a lazy run at the end of the pipeline")
            args, kwargs = self._lazy_run
            batch_res = self.next_batch(*args, **kwargs)
        elif True or kwargs.get('prefetch', 0) > 0: # FIXME
            if self._batch_generator is None:
                self._lazy_run = args, kwargs
                self._batch_generator = self.gen_batch(*args, **kwargs)
            batch_res = next(self._batch_generator)
        else:
            _kwargs = kwargs.copy()
            # target is not used here, but people tend to forget removing it when set prefetch to 0
            _kwargs.pop('target')
            # prefetch could be 0
            _kwargs.pop('prefetch')
            batch_res = None
            while batch_res is None:
                batch_index = self.index.next_batch(*args, **_kwargs)
                try:
                    batch_res = self.create_batch(batch_index, **_kwargs)
                except SkipBatchException:
                    pass
        self.elapsed_time += time.time() - start_time
        return batch_res

    def run(self, *args, **kwargs):
        """ Execute all lazy actions for each batch in the dataset

        See also
        --------
        :meth:`~.Pipeline.gen_batch`
        """
        if kwargs.pop('lazy', False):
            self._lazy_run = args, kwargs
        else:
            if self._lazy_run:
                _args, _kwargs = self._lazy_run
                args = _args if len(args) == 0 else args
                kwargs = {**_kwargs, **kwargs}
            if 'n_epochs' not in kwargs and 'n_iters' not in kwargs:
                kwargs['n_epochs'] = 1
            if 'n_epochs' in kwargs and kwargs['n_epochs'] is None:
                warnings.warn('Pipeline will never stop as n_epochs=None')

            self._batch_generator = self.gen_batch(*args, **kwargs)
            for _ in self._batch_generator:
                pass
            self._batch_generator = None

        return self

    def run_now(self, *args, **kwargs):
        """ Execute pipeline immediately """
        return self.run(*args, **kwargs, lazy=False)

    def run_later(self, *args, **kwargs):
        """ Define params to execute pipeline later """
        return self.run(*args, **kwargs, lazy=True)
