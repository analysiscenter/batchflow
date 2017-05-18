""" Pipeline decorators """
import os
import concurrent.futures as cf
import asyncio


def _cpu_count():
    cpu_count = 0
    try:
        cpu_count = len(os.sched_getaffinity(0))
    except AttributeError:
        cpu_count = os.cpu_count()
    return cpu_count


class model:
    """ Decorator for model definition methods in Batch classes """
    models = dict()

    def __init__(self, type='static', engine='tf'):
        self.type = type
        self.engine = engine
        self.method = None

    def run_model(self):
        """ Run and compile a model """
        get_tensor_name, input, model = self.method()
        model_spec = dict(get_tensor_name=get_tensor_name, input=input, model=model)
        models.update({self.method: model_spec})

    def __call__(self, method):
        self.method = method
        if self.type == 'static':
            self.run_model()

        def method_call(*args, **kwargs):
            """ Do nothing if the method is called explicitly """
            return None
        method_call.__model_method__ = method
        return method_call


class action_model:
    """ Decorator for Batch class actions based on a given model """
    def __init__(self, model):
        self.model_name = model

    def __call__(self, method):
        self.action = method

        def action_call(action_self, *args, **kwargs):
            if hasattr(action_self, self.model_name):
                try:
                    self.model_method = getattr(action_self, self.model_name).__model_method__
                except AttributeError:
                    raise ValueError("The method '%s' is not marked with @model" % self.model_name)
            else:
                raise ValueError("There is no such method '%s'" % self.model_name)

            model_params = model.models[self.model_method]
            return method(action_self, *args, **kwargs)
        method.action = True
        return action_call


def action(method):
    """ Decorator for action methods in Batch classes """
    # use __action for class-specific params
    method.action = True
    return method


def any_action_failed(results):
    """ Return True if some of the results come from failed Future """
    return any([isinstance(res, Exception) for res in results])


def inbatch_parallel(init, post=None, target='threads', **dec_kwargs):
    """ Make in-batch parallel decorator """
    if target not in ['nogil', 'threads', 'mpc', 'async']:
        raise ValueError("target should be one of 'nogil', threads', 'mpc', 'async'")

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
            if post_fn is None:
                # TODO: process errors in tasks
                return self
            else:
                all_results = []
                for future in futures:
                    try:
                        result = future.result()
                    except Exception as exce:  # pylint: disable=broad-except
                        result = exce
                    finally:
                        all_results += [result]
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

            n_workers = kwargs.pop('n_workers', _cpu_count())
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

            n_workers = kwargs.pop('n_workers', _cpu_count())
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

        def wrapped_method(self, *args, **kwargs):
            """ Wrap a method in a required parallel engine """
            if target == 'threads':
                return wrap_with_threads(self, args, kwargs)
            elif target == 'nogil':
                return wrap_with_threads(self, args, kwargs, nogil=True)
            elif target == 'mpc':
                return wrap_with_mpc(self, args, kwargs)
            elif target == 'async':
                return wrap_with_async(self, args, kwargs)
            raise ValueError('Wrong parallelization target:', target)
        return wrapped_method
    return inbatch_parallel_decorator
