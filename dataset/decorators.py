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


def action(method):
    """ Decorator for action methods in Batch classes """
    # use __action for class-specific params
    method.action = True
    return method


def inbatch_parallel(init, post=None, target='threads'):
    """ Make in-batch parallel decorator """
    if target not in ['nogil', 'threads', 'mpc', 'async', 'dd']:
        raise ValueError("target should one of 'nogil', threads', 'mpc', 'async', 'dd'")

    def inbatch_parallel_decorator(method):
        """ Return a decorator which run a method in parallel """
        def _check_functions(self):
            """ Check dcorator's `init` and `post` parameters """
            if init is None:
                raise ValueError("init cannot be None")
            else:
                init_fn = getattr(self, init)
                if not callable(init_fn):
                    raise ValueError("init should refer to a method of class", type(self).__name__,
                                     "returning the list of arguments")
            if post is not None:
                post_fn = getattr(self, post)
                if not callable(post_fn):
                    raise ValueError("post should refer to a method of class", type(self).__name__)
            else:
                post_fn = None
            return init_fn, post_fn

        def _make_args(args):
            """ Make args, kwargs tuple """
            if isinstance(args, tuple) and len(args) == 2:
                margs, mkwargs = args
            elif isinstance(args, dict):
                margs = []
                mkwargs = args
            else:
                mkwargs = dict()
                margs = args
            return margs, mkwargs

        def wrap_with_threads(self, args, kwargs, nogil=False):
            """ Run a method in parallel """
            init_fn, post_fn = _check_functions(self)

            n_workers = kwargs.get('n_workers', _cpu_count())
            with cf.ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = []
                if nogil:
                    nogil_fn = method(self)
                for arg in init_fn(self, *args, **kwargs):
                    margs, mkwargs = _make_args(arg)
                    if nogil:
                        one_ft = executor.submit(nogil_fn, *margs, **mkwargs)
                    else:
                        one_ft = executor.submit(method, self, *margs, **mkwargs)
                    futures.append(one_ft)
                timeout = kwargs.get('timeout', None)
                done, not_done = cf.wait(futures, timeout=timeout, return_when=cf.ALL_COMPLETED)

            if post_fn is None:
                return self
            else:
                done_results = [done_f.result() for done_f in done]
                return post_fn(done_results, not_done)

        def wrap_with_async(self, args, kwargs):
            """ Run a method in parallel with async / await """
            loop = kwargs.get('loop', asyncio.get_event_loop())
            init_fn, post_fn = _check_functions(self)

            futures = []
            for arg in init_fn(self, *args, **kwargs):
                margs, mkwargs = _make_args(arg)
                futures.append(method(self, *margs, **mkwargs))

            done = loop.run_until_complete(asyncio.gather(*futures, loop=loop))
            if post_fn is None:
                return self
            else:
                return post_fn(done)


        def wrapped_method(self, *args, **kwargs):
            """ Wrap a method in a required parallel engine """
            if target == 'threads':
                return wrap_with_threads(self, args, kwargs)
            elif target == 'nogil':
                return wrap_with_threads(self, args, kwargs, nogil=True)
            elif target == 'async':
                return wrap_with_async(self, args, kwargs)
            raise ValueError('Wrong parallelization target:', target)
        return wrapped_method
    return inbatch_parallel_decorator
