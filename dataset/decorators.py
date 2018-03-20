""" Pipeline decorators """
import os
import traceback
import threading
import concurrent.futures as cf
import asyncio
import functools
import logging
import inspect
try:
    from numba import jit
except ImportError:
    jit = None

from .named_expr import P


def _workers_count():
    cpu_count = 0
    try:
        cpu_count = len(os.sched_getaffinity(0))
    except AttributeError:
        cpu_count = os.cpu_count()
    return cpu_count * 4


def _make_action_wrapper_with_args(use_lock=None):    # pylint: disable=redefined-outer-name
    return functools.partial(_make_action_wrapper, _use_lock=use_lock)

def _make_action_wrapper(action_method, _use_lock=None):
    @functools.wraps(action_method)
    def _action_wrapper(action_self, *args, **kwargs):
        """ Call the action method """
        if _use_lock is not None:
            if action_self.pipeline is not None:
                if isinstance(_use_lock, bool):
                    _lock_name = '#_lock_' + action_method.__name__
                else:
                    _lock_name = _use_lock
                if not action_self.pipeline.has_variable(_lock_name):
                    action_self.pipeline.init_variable(_lock_name, threading.Lock())
                action_self.pipeline.get_variable(_lock_name).acquire()

        _res = action_method(action_self, *args, **kwargs)

        if _use_lock is not None:
            if action_self.pipeline is not None:
                action_self.pipeline.get_variable(_lock_name).release()

        return _res

    _action_wrapper.action = dict(method=action_method, use_lock=_use_lock)
    return _action_wrapper

def action(*args, **kwargs):
    """ Decorator for action methods in :class:`~dataset.Batch` classes

    Examples
    --------

    .. code-block:: python

        @action
        def some_action(self, arg1, arg2):
            ...

        @action(model='some_model')
        def train_model(self, model, another_arg):
            ...

        @action(use_lock=True)
        def critical_section(self, some_arg, another_arg):
            ...

        @action(use_lock='lock_name')
        def another_critical_section(self, some_arg, another_arg):
            ...
    """
    if len(args) == 1 and callable(args[0]):
        # action without arguments
        return _make_action_wrapper(action_method=args[0])
    # action with arguments
    return _make_action_wrapper_with_args(*args, **kwargs)


def any_action_failed(results):
    """ Return `True` if some parallelized invocations threw exceptions """
    return any(isinstance(res, Exception) for res in results)

def inbatch_parallel(init, post=None, target='threads', _use_self=True, **dec_kwargs):
    """ Decorator for parallel methods in :class:`~dataset.Batch` classes"""
    if target not in ['nogil', 'threads', 'mpc', 'async', 'for', 't', 'm', 'a', 'f']:
        raise ValueError("target should be one of 'threads', 'mpc', 'async', 'for'")

    def inbatch_parallel_decorator(method):
        """ Return a decorator which run a method in parallel """
        def _check_functions(self):
            """ Check dcorator's `init` and `post` parameters """
            if init is None:
                raise ValueError("init cannot be None")
            else:
                if isinstance(init, str):
                    try:
                        init_fn = getattr(self, init)
                    except AttributeError:
                        raise ValueError("init should refer to a method or property of the class", type(self).__name__,
                                         "returning the list of arguments")
                elif callable(init):
                    init_fn = init
                else:
                    init_fn = lambda *a, **k: init

            if post is not None:
                if isinstance(init, str):
                    try:
                        post_fn = getattr(self, post)
                    except AttributeError:
                        raise ValueError("post should refer to a method of the class", type(self).__name__)
                elif callable(post):
                    post_fn = post
                else:
                    post_fn = lambda *a, **k: post
                if not callable(post_fn):
                    raise ValueError("post should refer to a callable or a method of the batch class")
            else:
                post_fn = None
            return init_fn, post_fn

        def _call_init_fn(init_fn, args, kwargs):
            if callable(init_fn):
                return init_fn(*args, **kwargs)
            return init_fn

        def _call_post_fn(self, post_fn, futures, args, kwargs):
            all_results = []
            for future in futures:
                try:
                    if isinstance(future, (cf.Future, asyncio.Task)):
                        result = future.result()
                    else:
                        result = future
                except Exception as exce:  # pylint: disable=broad-except
                    result = exce
                finally:
                    all_results += [result]

            if post_fn is None:
                if any_action_failed(all_results):
                    all_errors = [error for error in all_results if isinstance(error, Exception)]
                    print(all_errors)
                    traceback.print_tb(all_errors[0].__traceback__)
                return self
            else:
                return post_fn(all_results, *args, **kwargs)

        def _prepare_args(self, args, kwargs):
            params = list()

            def _get_value(value, pos=None, name=None):
                if isinstance(value, P):
                    if pos is not None:
                        params.append(pos)
                    elif name is not None:
                        params.append(name)
                    v = value.get(batch=self, parallel=True)
                    return v
                return value

            _args = []
            for i, v in enumerate(args):
                _args.append(_get_value(v, pos=i))
            _kwargs = {}
            for k, v in kwargs.items():
                _kwargs.update({k: _get_value(v, name=k)})

            return _args, _kwargs, params

        def _make_args(self, iteration, init_args, args, kwargs, params=None):
            """ Make args, kwargs tuple """
            if isinstance(init_args, tuple) and len(init_args) == 2:
                margs, mkwargs = init_args
            elif isinstance(init_args, dict):
                margs = list()
                mkwargs = init_args
            else:
                margs = init_args
                mkwargs = dict()
            margs = margs if isinstance(margs, (list, tuple)) else [margs]

            if params:
                _args = list(args)
                _kwargs = {**kwargs}
                for k in params:
                    if isinstance(k, str):
                        _kwargs[k] = _kwargs[k][iteration]
                    else:
                        _args[k] = _args[k][iteration]
            else:
                _args = args
                _kwargs = kwargs

            if len(args) > 0:
                margs = list(margs) + list(_args)
            if len(kwargs) > 0:
                mkwargs.update(_kwargs)

            if _use_self and self is not None:
                margs = [self] + margs

            return margs, mkwargs

        def wrap_with_threads(self, args, kwargs):
            """ Run a method in parallel """
            init_fn, post_fn = _check_functions(self)

            n_workers = kwargs.pop('n_workers', _workers_count())
            with cf.ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = []
                args, kwargs, params = _prepare_args(self, args, kwargs)
                full_kwargs = {**dec_kwargs, **kwargs}
                for iteration, arg in enumerate(_call_init_fn(init_fn, args, full_kwargs)):
                    margs, mkwargs = _make_args(self, iteration, arg, args, kwargs, params)
                    one_ft = executor.submit(method, *margs, **mkwargs)
                    futures.append(one_ft)

                timeout = kwargs.get('timeout', None)
                cf.wait(futures, timeout=timeout, return_when=cf.ALL_COMPLETED)

            return _call_post_fn(self, post_fn, futures, args, full_kwargs)

        def wrap_with_mpc(self, args, kwargs):
            """ Run a method in parallel """
            init_fn, post_fn = _check_functions(self)

            n_workers = kwargs.pop('n_workers', _workers_count())
            with cf.ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = []
                mpc_func = method(self, *args, **kwargs)
                args, kwargs, params = _prepare_args(self, args, kwargs)
                full_kwargs = {**dec_kwargs, **kwargs}
                for iteration, arg in enumerate(_call_init_fn(init_fn, args, full_kwargs)):
                    margs, mkwargs = _make_args(None, iteration, arg, args, kwargs, params)
                    one_ft = executor.submit(mpc_func, *margs, **mkwargs)
                    futures.append(one_ft)

                timeout = kwargs.pop('timeout', None)
                cf.wait(futures, timeout=timeout, return_when=cf.ALL_COMPLETED)

            return _call_post_fn(self, post_fn, futures, args, full_kwargs)

        def wrap_with_async(self, args, kwargs):
            """ Run a method in parallel with async / await """
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                # this is a new thread where there is no loop
                loop = kwargs.get('loop', None)
                asyncio.set_event_loop(loop)
            else:
                loop = kwargs.get('loop', loop)

            init_fn, post_fn = _check_functions(self)

            futures = []
            args, kwargs, params = _prepare_args(self, args, kwargs)
            full_kwargs = {**dec_kwargs, **kwargs}
            for iteration, arg in enumerate(_call_init_fn(init_fn, args, full_kwargs)):
                margs, mkwargs = _make_args(self, iteration, arg, args, kwargs, params)
                futures.append(asyncio.ensure_future(method(*margs, **mkwargs)))

            loop.run_until_complete(asyncio.gather(*futures, loop=loop, return_exceptions=True))

            return _call_post_fn(self, post_fn, futures, args, full_kwargs)

        def wrap_with_for(self, args, kwargs):
            """ Run a method sequentially (without parallelism) """
            init_fn, post_fn = _check_functions(self)

            _ = kwargs.pop('n_workers', _workers_count())
            futures = []
            args, kwargs, params = _prepare_args(self, args, kwargs)
            full_kwargs = {**dec_kwargs, **kwargs}
            for iteration, arg in enumerate(_call_init_fn(init_fn, args, full_kwargs)):
                margs, mkwargs = _make_args(self, iteration, arg, args, kwargs, params)
                try:
                    one_ft = method(*margs, **mkwargs)
                except Exception as e:   # pylint: disable=broad-except
                    one_ft = e
                futures.append(one_ft)

            return _call_post_fn(self, post_fn, futures, args, full_kwargs)

        @functools.wraps(method)
        def wrapped_method(self, *args, **kwargs):
            """ Wrap a method with a required parallel engine """
            if not _use_self:
                # when use_self=False, the first arg is not self, but an ordinary arg
                args = (self,) + args
                self = None
            if 'target' in kwargs:
                _target = kwargs.pop('target')
            else:
                _target = target

            if asyncio.iscoroutinefunction(method) or _target in ['async', 'a']:
                return wrap_with_async(self, args, kwargs)
            elif _target in ['threads', 't']:
                return wrap_with_threads(self, args, kwargs)
            elif _target in ['mpc', 'm']:
                return wrap_with_mpc(self, args, kwargs)
            elif _target in ['for', 'f']:
                return wrap_with_for(self, args, kwargs)
            raise ValueError('Wrong parallelization target:', _target)
        return wrapped_method
    return inbatch_parallel_decorator



def parallel(*args, _use_self=False, **kwargs):
    """ Decorator for a parallel execution of a function """
    return inbatch_parallel(*args, _use_self=_use_self, **kwargs)


def njit(nogil=True):
    """ Fake njit decorator to use when numba is not installed """
    _ = nogil
    def njit_fake_decorator(method):
        """ Return a decorator """
        @functools.wraps(method)
        def wrapped_method(*args, **kwargs):
            """ Log warning that numba is not installed which causes preformance degradation """
            logging.warning('numba is not installed. This causes a severe performance degradation for method %s',
                            method.__name__)
            return method(*args, **kwargs)
        return wrapped_method
    return njit_fake_decorator


def mjit(*args, nopython=True, nogil=True, **kwargs):
    """ jit decorator for methods """
    def _jit(method):
        source = inspect.getsource(method).split('\n')
        indent = len(source[0]) - len(source[0].lstrip())
        source = [s[indent:] for s in source if len(s) >= indent and s[indent] != '@']
        source = '\n'.join(source)
        globs = method.__globals__.copy()
        exec(source, globs)  # pylint: disable=exec-used
        if jit is not None:
            func = jit(*args, nopython=nopython, nogil=nogil, **kwargs)(globs[method.__name__])
        else:
            func = method
            logging.warning('numba is not installed. This causes a severe performance degradation for method %s',
                            method.__name__)

        @functools.wraps(method)
        def _wrapped_method(self, *args, **kwargs):
            res = func(None, *args, **kwargs) or self
            return res
        return _wrapped_method

    if len(args) == 1 and (callable(args[0])) and len(kwargs) == 0:
        method = args[0]
        args = tuple()
        return _jit(method)
    return _jit
