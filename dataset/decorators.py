""" Pipeline decorators """
import os
import inspect
import traceback
import threading
import concurrent.futures as cf
import asyncio
import functools


def _workers_count():
    cpu_count = 0
    try:
        cpu_count = len(os.sched_getaffinity(0))
    except AttributeError:
        cpu_count = os.cpu_count()
    return cpu_count * 4


def make_method_key2(module_name, qual_name):
    """ Build a full method name 'module.method' """
    return module_name + '.' + qual_name

def make_method_key3(module_name, class_name, method_name):
    """ Build a full method name 'module.method' """
    return make_method_key2(module_name, class_name + '.' + method_name)

def get_method_key(method):
    """ Retrieve a full method name from a callable """
    return make_method_key2(inspect.getmodule(method).__name__, method.__qualname__)

def infer_method_key(action_method, model_name):
    """ Infer a full model method name from a given action method and a model name """
    return make_method_key3(inspect.getmodule(action_method).__name__,
                            action_method.__qualname__.rsplit('.', 1)[0], model_name)

def infer_bound_method_key(instance_or_method, model_name):
    """ Infer a full model method name from a given bound method and a model name """
    if hasattr(instance_or_method, '__self__'):
        instance = instance_or_method.__self__
    else:
        instance = instance_or_method
    return make_method_key3(inspect.getmodule(instance).__name__,
                            instance.__class__.__name__, model_name)


class ModelDecorator:
    """ Decorator for model definition methods in Batch classes """
    models = dict()

    def __init__(self, mode='global', engine='tf'):
        self.mode = mode
        self.engine = engine
        self.method = None

    @staticmethod
    def get_model(method):
        """ Return a model specification for a given model method """
        full_method_name = get_method_key(method)
        return ModelDecorator.models[full_method_name]

    @staticmethod
    def get_model_by_name(instance, model_name):
        """ Return a model specification given its name """
        # method = getattr(instance, model_name)
        full_model_name = infer_bound_method_key(instance, model_name)
        return ModelDecorator.models[full_model_name]

    @staticmethod
    def get_all_model_names(instance):
        """ Return all model names for a given batch instance """
        full_model_pattern = infer_bound_method_key(instance, '')
        all_models = [m.rsplit('.')[-1] for m in ModelDecorator.models if full_model_pattern in m]
        return all_models

    @staticmethod
    def add_model(method, model_spec):
        """ Add a model specification into the model directory """
        full_method_name = get_method_key(method)
        ModelDecorator.models.update({full_method_name: model_spec})

    def run_model(self):
        """ Run the model method and save the model into the model directory """
        model_spec = self.method()
        self.add_model(self.method, model_spec)

    def __call__(self, method):
        self.method = method
        if self.mode == 'global':
            self.run_model()

        def method_call(*args, **kwargs):
            """ Do nothing if the method is called explicitly """
            _ = args, kwargs
            return None
        method_call.model_method = self.method
        return method_call

def model(*args, **kwargs):
    """ Decorator for model methods

    Usage:
        @model()
        def some_model():
            ...
            return my_model
    """
    return ModelDecorator(*args, **kwargs)


def _make_action_wrapper_with_args(model=None, singleton=False):
    return functools.partial(_make_action_wrapper, _model_name=model, _singleton=singleton)

def _make_action_wrapper(action_method, _model_name=None, _singleton=False):
    _singleton_lock = None if not _singleton else threading.Lock()

    @functools.wraps(action_method)
    def action_wrapper(action_self, *args, **kwargs):
        """ Call the action method """
        if _singleton_lock is not None:
            _singleton_lock.acquire(blocking=True)

        if _model_name is None:
            _res = action_method(action_self, *args, **kwargs)
        else:
            if hasattr(action_self, _model_name):
                try:
                    _model_method = getattr(action_self, _model_name).model_method
                except AttributeError:
                    raise ValueError("The method '%s' is not marked with @model" % _model_name)
            else:
                raise ValueError("There is no such method '%s'" % _model_name)

            _model_spec = ModelDecorator.get_model(_model_method)
            _res = action_method(action_self, _model_spec, *args, **kwargs)

        if _singleton_lock is not None:
            _singleton_lock.release()
        return _res

    action_wrapper.action = True
    return action_wrapper

def action(*args, **kwargs):
    """ Decorator for action methods in Batch classes

    Usage:
        @action
        def some_action(self, arg1, arg2):
            ...

        @action(model='some_model')
        def train_model(self, model, another_arg):
            ...
    """
    if len(args) == 1 and callable(args[0]):
        # action without arguments
        return _make_action_wrapper(action_method=args[0])
    else:
        # action with arguments
        return _make_action_wrapper_with_args(*args, **kwargs)


def any_action_failed(results):
    """ Return True if some parallelized invocations threw exceptions """
    return any(isinstance(res, Exception) for res in results)

def inbatch_parallel(init, post=None, target='threads', **dec_kwargs):
    """ Make in-batch parallel decorator """
    if target not in ['nogil', 'threads', 'mpc', 'async', 'for', 't', 'm', 'a', 'f']:
        raise ValueError("target should be one of 'threads', 'mpc', 'async', 'for'")

    def inbatch_parallel_decorator(method):
        """ Return a decorator which run a method in parallel """
        def _check_functions(self):
            """ Check dcorator's `init` and `post` parameters """
            if init is None:
                raise ValueError("init cannot be None")
            else:
                try:
                    init_fn = getattr(self, init)
                except AttributeError:
                    raise ValueError("init should refer to a method or property of the class", type(self).__name__,
                                     "returning the list of arguments")
            if post is not None:
                try:
                    post_fn = getattr(self, post)
                except AttributeError:
                    raise ValueError("post should refer to a method of the class", type(self).__name__)
                if not callable(post_fn):
                    raise ValueError("post should refer to a method of the class", type(self).__name__)
            else:
                post_fn = None
            return init_fn, post_fn

        def _call_init_fn(init_fn, args, kwargs):
            if callable(init_fn):
                return init_fn(*args, **kwargs)
            else:
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

        def _make_args(init_args, args, kwargs):
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
            if len(args) > 0:
                margs = list(margs) + list(args)
            if len(kwargs) > 0:
                mkwargs.update(kwargs)
            return margs, mkwargs

        def wrap_with_threads(self, args, kwargs, nogil=False):
            """ Run a method in parallel """
            init_fn, post_fn = _check_functions(self)

            n_workers = kwargs.pop('n_workers', _workers_count())
            with cf.ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = []
                if nogil:
                    nogil_fn = method(self, *args, **kwargs)
                full_kwargs = {**kwargs, **dec_kwargs}
                for arg in _call_init_fn(init_fn, args, full_kwargs):
                    margs, mkwargs = _make_args(arg, args, kwargs)
                    if nogil:
                        one_ft = executor.submit(nogil_fn, *margs, **mkwargs)
                    else:
                        one_ft = executor.submit(method, self, *margs, **mkwargs)
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
                full_kwargs = {**kwargs, **dec_kwargs}
                for arg in _call_init_fn(init_fn, args, full_kwargs):
                    margs, mkwargs = _make_args(arg, args, kwargs)
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
            full_kwargs = {**kwargs, **dec_kwargs}
            for arg in _call_init_fn(init_fn, args, full_kwargs):
                margs, mkwargs = _make_args(arg, args, kwargs)
                futures.append(asyncio.ensure_future(method(self, *margs, **mkwargs)))

            loop.run_until_complete(asyncio.gather(*futures, loop=loop, return_exceptions=True))

            return _call_post_fn(self, post_fn, futures, args, full_kwargs)

        def wrap_with_for(self, args, kwargs):
            """ Run a method in parallel """
            init_fn, post_fn = _check_functions(self)

            _ = kwargs.pop('n_workers', _workers_count())
            futures = []
            full_kwargs = {**kwargs, **dec_kwargs}
            for arg in _call_init_fn(init_fn, args, full_kwargs):
                margs, mkwargs = _make_args(arg, args, kwargs)
                try:
                    one_ft = method(self, *margs, **mkwargs)
                    if callable(one_ft):
                        one_ft = one_ft(*margs, **mkwargs)
                except Exception as e:   # pylint: disable=broad-except
                    one_ft = e
                futures.append(one_ft)

            return _call_post_fn(self, post_fn, futures, args, full_kwargs)

        @functools.wraps(method)
        def wrapped_method(self, *args, **kwargs):
            """ Wrap a method in a required parallel engine """
            if asyncio.iscoroutinefunction(method) or target == 'async':
                return wrap_with_async(self, args, kwargs)
            if target in ['threads', 't']:
                return wrap_with_threads(self, args, kwargs)
            elif target == 'nogil':
                return wrap_with_threads(self, args, kwargs, nogil=True)
            elif target in ['mpc', 'm']:
                return wrap_with_mpc(self, args, kwargs)
            elif target in ['for', 'f']:
                return wrap_with_for(self, args, kwargs)
            raise ValueError('Wrong parallelization target:', target)
        return wrapped_method
    return inbatch_parallel_decorator


parallel = inbatch_parallel  # pylint: disable=invalid-name
