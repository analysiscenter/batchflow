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
from .utils_random import make_seed_sequence, spawn_seed_sequence


def make_function(method, is_global=False):
    """ Makes a function from a method

    Parameters
    ----------
    method
        a callable

    is_global : bool
        whether to create a function in a global namespace

    Notes
    -----
    A method should not be decorated with any other decorator.
    """
    source = inspect.getsource(method).split('\n')
    indent = len(source[0]) - len(source[0].lstrip())

    # strip indent spaces
    source = [s[indent:] for s in source if len(s) > indent]
    # skip all decorator and comment lines before 'def' or 'async def'
    start = 0
    for i, s in enumerate(source):
        if s[:3] in ['def', 'asy']:
            start = i
            break
    source = '\n'.join(source[start:])

    globs = globals() if is_global else method.__globals__.copy()
    exec(source, globs)    # pylint:disable=exec-used

    # Method with the same name might exist in various classes or modules
    # so a global function should have a unique name
    function_name = method.__module__ + "_" + method.__qualname__
    function_name = function_name.replace('.', '_')
    globs[function_name] = globs[method.__name__]
    return globs[function_name]


def _workers_count():
    cpu_count = 0
    try:
        cpu_count = len(os.sched_getaffinity(0))
    except AttributeError:
        cpu_count = os.cpu_count()
    return cpu_count * 4


def _make_action_wrapper_with_args(use_lock=None, no_eval=None):    # pylint: disable=redefined-outer-name
    return functools.partial(_make_action_wrapper, use_lock=use_lock, no_eval=no_eval)

def _make_action_wrapper(action_method, use_lock=None, no_eval=None):
    @functools.wraps(action_method)
    def _action_wrapper(action_self, *args, **kwargs):
        """ Call the action method """
        if use_lock is not None:
            if action_self.pipeline is not None:
                if isinstance(use_lock, bool):
                    _lock_name = '#_lock_' + action_method.__name__
                else:
                    _lock_name = use_lock
                if not action_self.pipeline.has_variable(_lock_name):
                    action_self.pipeline.init_variable(_lock_name, threading.Lock())
                action_self.pipeline.get_variable(_lock_name).acquire()

        _res = action_method(action_self, *args, **kwargs)

        if use_lock is not None:
            if action_self.pipeline is not None:
                action_self.pipeline.get_variable(_lock_name).release()

        return _res

    if isinstance(no_eval, str):
        no_eval = [no_eval]
    _action_wrapper.action = dict(method=action_method, use_lock=use_lock, no_eval=no_eval)
    return _action_wrapper

def action(*args, **kwargs):
    """ Decorator for action methods in :class:`~.Batch` classes

    Parameters
    ----------
    use_lock : bool or str
        whether to lock an action when a pipeline is executed. It can be bool or a lock name.
        A pipeline variable with a lock is created in the pipeline during the execution.

    no_eval : str or a sequence of str
        parameters to skip from named expression evaluation.
        A parameter should be passed as a named argument only.

    Examples
    --------

    .. code-block:: python

        @action
        def some_action(self, arg1, arg2):
            ...

        @action(no_eval='dst')
        def calc_offset(self, src, dst=None):
            ...

        @action(use_lock=True)
        def critical_section(self, some_arg, another_arg):
            ...

        @action(use_lock='lock_name')
        def another_critical_section(self, some_arg, another_arg):
            ...
    """
    if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
        # action without arguments
        return _make_action_wrapper(action_method=args[0])
    # action with arguments
    return _make_action_wrapper_with_args(*args, **kwargs)


def apply_parallel(*args, **kwargs):
    """ Mark class method for transform in its metaclass.

        Decorator writes `kwargs` to the method attribute `apply_kwargs`,
        so they can be extracted and used in metaclass.

        Parameters
        ----------
        args, kwargs
            other parameters passed to `apply_parallel` method of the class
            where this decorator is being used

        Notes
        -----
        Redefine the attribute `apply_defaults <.Batch.apply_defaults>` in
        the batch class. This is proposed solely for the purposes of brevity — in
        order to avoid repeated heavily loaded class methods decoration, e.g.
        `@apply_parallel(src='images', target='for')` which in most cases is
        actually equivalent to simple `@apply_parallel` assuming
        that the defaults are redefined for the class whose methods are being
        transformed.

        Note, that if no defaults redefined those from the nearest
        parent class will be used in :class:`~.batch.MethodsTransformingMeta`.
        """
    def mark(method):
        method.apply_kwargs = kwargs
        return method

    if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
        return mark(args[0])
    if len(args) != 0:
        raise ValueError("This decorator accepts only named arguments")

    return mark


def any_action_failed(results):
    """ Return `True` if some parallelized invocations threw exceptions """
    return any(isinstance(res, Exception) for res in results)

def call_method(method, use_self, args, kwargs, seed=None):
    """ Call a method with given args """
    if use_self and hasattr(args[0], 'random_seed') and seed is not None:
        # set batch.random_seed to create RNG
        args[0].random_seed = seed
    return method(*args, **kwargs)

def inbatch_parallel(init, post=None, target='threads', _use_self=None, **dec_kwargs):
    """ Decorator for parallel methods in :class:`~.Batch` classes

    Parameters
    ----------
    init
        a method name or a callable that returns an iterable for parallelization
        (e.g. a list of indices or items to be passed to a parallelized method)
    post
        a method name or a callable to call after parallel invocations
        (e.g. to assemble the batch)
    target : 'threads', 'mpc', 'async', 'for'
        a parallelization engine
    _use_self : bool
        whether to pass `self` (i.e. whether a decorated callable is a method or a function)

    Notes
    -----
    `mpc` can be used with a method that is decorated only by `inbatch_parallel`.
    All other decorators will be ignored.
    """
    if target not in ['nogil', 'threads', 'mpc', 'async', 'for', 't', 'm', 'a', 'f']:
        raise ValueError("target should be one of 'threads', 'mpc', 'async', 'for'")

    def inbatch_parallel_decorator(method):
        """ Return a decorator which run a method in parallel """
        use_self = '.' in method.__qualname__ if _use_self is None else _use_self
        mpc_method = method
        if use_self:
            try:
                mpc_method = make_function(method, is_global=True)
            except Exception:  # pylint:disable=broad-except
                mpc_method = None

        def _check_functions(self):
            """ Check decorator's `init` and `post` parameters """
            if init is None:
                raise ValueError("init cannot be None")

            if isinstance(init, str):
                try:
                    init_fn = getattr(self, init)
                except AttributeError as e:
                    raise ValueError("init should refer to a method or property of the class", type(self).__name__,
                                     "returning the list of arguments") from e
            elif callable(init):
                init_fn = init
            else:
                init_fn = init

            if isinstance(post, str):
                try:
                    post_fn = getattr(self, post)
                except AttributeError as e:
                    raise ValueError("post should refer to a method of the class", type(self).__name__) from e
            elif callable(post):
                post_fn = post
            else:
                post_fn = post

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
                    logging.error("Parallel action failed %s", all_errors)
                    traceback.print_tb(all_errors[0].__traceback__)
                    raise RuntimeError("Parallel action failed")
                return self
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
            if isinstance(init_args, tuple) and len(init_args) == 2 and \
               isinstance(init_args[0], tuple) and isinstance(init_args[1], dict):
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

            if use_self:
                margs = [self] + list(margs)

            return margs, mkwargs

        def make_random_seed(self):
            if getattr(self, 'random_state', None) is None:
                return make_seed_sequence()
            return self.random_stat

        def wrap_with_threads(self, args, kwargs):
            """ Run a method in parallel threads """
            init_fn, post_fn = _check_functions(self)

            n_workers = kwargs.pop('n_workers', _workers_count())
            with cf.ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = []
                args, kwargs, params = _prepare_args(self, args, kwargs)
                full_kwargs = {**dec_kwargs, **kwargs}
                for iteration, arg in enumerate(_call_init_fn(init_fn, args, full_kwargs)):
                    margs, mkwargs = _make_args(self, iteration, arg, args, kwargs, params)
                    seed = None if getattr(self, 'random_state', None) is None else spawn_seed_sequence(self)
                    one_ft = executor.submit(call_method, method, use_self, margs, mkwargs, seed=seed)
                    futures.append(one_ft)

                timeout = kwargs.get('timeout', None)
                cf.wait(futures, timeout=timeout, return_when=cf.ALL_COMPLETED)

            return _call_post_fn(self, post_fn, futures, args, full_kwargs)

        def wrap_with_mpc(self, args, kwargs):
            """ Run a method in parallel processes """
            init_fn, post_fn = _check_functions(self)

            n_workers = kwargs.pop('n_workers', _workers_count())
            with cf.ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = []
                args, kwargs, params = _prepare_args(self, args, kwargs)
                full_kwargs = {**dec_kwargs, **kwargs}
                for iteration, arg in enumerate(_call_init_fn(init_fn, args, full_kwargs)):
                    margs, mkwargs = _make_args(self, iteration, arg, args, kwargs, params)
                    seed = None if getattr(self, 'random_state', None) is None else spawn_seed_sequence(self)
                    one_ft = executor.submit(call_method, mpc_method, use_self, margs, mkwargs, seed=seed)
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
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            else:
                # allow to specify a loop as an action parameter
                loop = kwargs.get('loop', loop)

            if loop.is_running():
                raise RuntimeError('Cannot parallel async methods with a running event loop (e.g. in IPython).')

            init_fn, post_fn = _check_functions(self)

            futures = []
            args, kwargs, params = _prepare_args(self, args, kwargs)
            full_kwargs = {**dec_kwargs, **kwargs}
            # save an initial seed to generate child seeds from
            random_seed = make_random_seed(self)
            for iteration, arg in enumerate(_call_init_fn(init_fn, args, full_kwargs)):
                margs, mkwargs = _make_args(self, iteration, arg, args, kwargs, params)
                seed = spawn_seed_sequence(random_seed)
                futures.append(loop.create_task(call_method(method, use_self, margs, mkwargs, seed=seed)))

            loop.run_until_complete(asyncio.gather(*futures, loop=loop, return_exceptions=True))

            return _call_post_fn(self, post_fn, futures, args, full_kwargs)

        def wrap_with_for(self, args, kwargs):
            """ Run a method sequentially (without parallelism) """
            init_fn, post_fn = _check_functions(self)
            _ = kwargs.pop('n_workers', _workers_count())
            futures = []
            args, kwargs, params = _prepare_args(self, args, kwargs)
            full_kwargs = {**dec_kwargs, **kwargs}
            # save an initial seed to generate child seeds from
            random_seed = make_random_seed(self)
            for iteration, arg in enumerate(_call_init_fn(init_fn, args, full_kwargs)):
                margs, mkwargs = _make_args(self, iteration, arg, args, kwargs, params)

                seed = spawn_seed_sequence(random_seed)
                try:
                    one_ft = call_method(method, use_self, margs, mkwargs, seed=seed)
                except Exception as e:   # pylint: disable=broad-except
                    one_ft = e
                futures.append(one_ft)

            return _call_post_fn(self, post_fn, futures, args, full_kwargs)

        @functools.wraps(method)
        def wrapped_method(*args, **kwargs):
            """ Wrap a method with a required parallel engine """
            if use_self:
                # the first arg is self, not an ordinary arg
                self = args[0]
                args = args[1:]
            else:
                # still need self to preserve the signatures of other functions
                self = None

            _target = kwargs.pop('target', target)

            if asyncio.iscoroutinefunction(method) or _target in ['async', 'a']:
                x = wrap_with_async(self, args, kwargs)
            elif _target in ['threads', 't']:
                x = wrap_with_threads(self, args, kwargs)
            elif _target in ['mpc', 'm']:
                if mpc_method is not None:
                    x = wrap_with_mpc(self, args, kwargs)
                else:
                    raise ValueError('Cannot use MPC with this method', method)
            elif _target in ['for', 'f']:
                x = wrap_with_for(self, args, kwargs)
            else:
                raise ValueError('Wrong parallelization target:', _target)
            return x
        return wrapped_method

    return inbatch_parallel_decorator



def parallel(*args, use_self=None, **kwargs):
    """ Decorator for a parallel execution of a function """
    return inbatch_parallel(*args, _use_self=use_self, **kwargs)


def njit(nogil=True, parallel=True):  # pylint: disable=redefined-outer-name
    """ Fake njit decorator to use when numba is not installed """
    _, _ = nogil, parallel
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
    """ jit decorator for methods

    Notes
    -----
    This decorator should be applied directly to a method, not another decorator.
    """
    def _jit(method):
        if jit is not None:
            func = make_function(method)
            func = jit(*args, nopython=nopython, nogil=nogil, **kwargs)(func)
        else:
            func = method
            logging.warning('numba is not installed. This causes a severe performance degradation for method %s',
                            method.__name__)

        @functools.wraps(method)
        def _wrapped_method(self, *args, **kwargs):
            _ = self
            return func(None, *args, **kwargs)
        return _wrapped_method

    if len(args) == 1 and (callable(args[0])) and len(kwargs) == 0:
        method = args[0]
        args = tuple()
        return _jit(method)
    return _jit


def deprecated(msg):
    """ Decorator for deprecated functions and methods """
    def decorator(func):
        @functools.wraps(func)
        def _call(*args, **kwargs):
            logging.warning(msg)
            return func(*args, **kwargs)
        return _call
    return decorator
