""" Contains basic Batch classes """
# pylint: disable=ungrouped-imports
import os
import traceback
import threading
import warnings
import functools

import dill
try:
    import blosc
except ImportError:
    pass
import numpy as np
try:
    import pandas as pd
except ImportError:
    from . import _fake as pd
try:
    import feather
except ImportError:
    pass
try:
    import dask.dataframe as dd
except ImportError:
    from . import _fake as dd

from .dsindex import DatasetIndex, FilesIndex
# renaming apply_parallel decorator is needed as Batch.apply_parallel method is also in the same namespace
# and can serve as a decorator too
from .decorators import action, inbatch_parallel, any_action_failed, apply_parallel as apply_parallel_
from .components import create_item_class, BaseComponents
from .named_expr import P, R
from .utils_random import make_rng


class MethodsTransformingMeta(type):
    """ A metaclass to transform all class methods in the way described below:

        1. Methods decorated with `@apply_parallel` are wrapped with `apply_parallel` method.

        2. Add the original version of the method (i.e. unwrapped) to a class
           namespace using name with underscores: `'_{}_'.format(name)`. This
           is necessary in order to allow inner calls of untransformed versions
           (e.g. `ImagesBatch.scale` calls `ImagesBatch.crop` under the hood).
    """
    def __new__(cls, name, bases, namespace):
        namespace_ = namespace.copy()
        for object_name, object_ in namespace.items():
            transform_kwargs = getattr(object_, 'apply_kwargs', None)
            if transform_kwargs is not None:
                namespace_[object_name] = cls.use_apply_parallel(object_, **transform_kwargs)

                disclaimer = f"This is an untransformed version of `{object_.__qualname__}`.\n\n"
                object_.__doc__ = disclaimer + (object_.__doc__ or '')
                object_.__name__ = '_' + object_name + '_'
                object_.__qualname__ = '.'.join(object_.__qualname__.split('.')[:-1] + [object_.__name__])
                namespace_[object_.__name__] = object_

        return super().__new__(cls, name, bases, namespace_)

    @classmethod
    def use_apply_parallel(cls, method, **apply_kwargs):
        """ Wrap passed `method` in accordance with `all` arg value """
        @functools.wraps(method)
        def apply_parallel_wrapper(self, *args, **kwargs):
            transform = self.apply_parallel
            method_ = method.__get__(self, type(self)) # bound method to class
            kwargs_full = {**self.apply_defaults, **apply_kwargs, **kwargs}
            return transform(method_, *args, **kwargs_full)
        return action(apply_parallel_wrapper)


class Batch(metaclass=MethodsTransformingMeta):
    """ The core Batch class

    Note, that if any method is wrapped with `@apply_parallel` decorator
    than for inner calls (i.e. from other methods) should be used version
    of desired method with underscores. (For example, if there is a decorated
    `method` than you need to call `_method_` from inside of `other_method`).
    Same is applicable for all child classes of :class:`batch.Batch`.
    """
    components = None
    # Class-specific defaults for :meth:`.Batch.apply_parallel`
    apply_defaults = dict(target='threads',
                          post='_assemble',
                          src=None,
                          dst=None,
                          )

    def __init__(self, index, dataset=None, pipeline=None, preloaded=None, copy=False, *args, **kwargs):
        _ = args
        if  self.components is not None and not isinstance(self.components, tuple):
            raise TypeError("components should be a tuple of strings with components names")
        self.index = index
        self._preloaded_lock = threading.Lock()
        self._preloaded = preloaded
        self._copy = copy
        self._local = threading.local()
        self._data_named = None
        self._data = None
        self._dataset = dataset
        self.pipeline = pipeline
        self.iteration = None
        self._attrs = None
        self.create_attrs(**kwargs)

    def create_attrs(self, **kwargs):
        """ Create attributes from kwargs """
        self._attrs = list(kwargs.keys())
        for attr, value in kwargs.items():
            setattr(self, attr, value)

    def get_attrs(self):
        """ Return additional attrs as kwargs """
        if self._attrs is None:
            return {}
        return {attr: getattr(self, attr, None) for attr in self._attrs}

    @property
    def data(self):
        """: tuple or named components - batch data """
        try:
            if self._data is None and self._preloaded is not None:
                # load data the first time it's requested
                with self._preloaded_lock:
                    if self._data is None and self._preloaded is not None:
                        self.load(src=self._preloaded)
            res = self._data if self.components is None else self._data_named
        except Exception as exc:
            print("Exception:", exc)
            traceback.print_tb(exc.__traceback__)
            raise
        return res

    @data.setter
    def data_setter(self, value):
        """: tuple or named components - batch data """
        self._data = value

    @property
    def dataset(self):
        """: Dataset - a dataset the batch has been taken from """
        if self.pipeline is not None:
            return self.pipeline.dataset
        return self._dataset

    @property
    def pipeline(self):
        """: Pipeline - a pipeline the batch is being used in """
        return self._local.pipeline

    @pipeline.setter
    def pipeline(self, value):
        """ Store the pipeline in a thread-local storage """
        self._local.pipeline = value

    @property
    def random(self):
        """ A random number generator :class:`numpy.random.Generator`.
        Use it instead of `np.random` for reproducibility.

        Examples
        --------

        ::

            x = self.random.normal(0, 1)
        """
        # if RNG is set for the batch (e.g. in @inbatch_parallel), use it
        if hasattr(self._local, 'random'):
            return self._local.random
        # otherwise use RNG from the pipeline
        if self.pipeline is not None and self.pipeline.random is not None:
            return self.pipeline.random

        # if there is none (e.g. when the batch is created manually), make a random one
        self._local.random = make_rng(self.random_seed)
        return self._local.random

    @property
    def random_seed(self):
        """ : SeedSequence for random number generation """
        # if RNG is set for the batch (e.g. in @inbatch_parallel), use it
        if hasattr(self._local, 'random_seed'):
            return self._local.random_seed

        if self.pipeline is not None and self.pipeline.random_seed is not None:
            return self.pipeline.random_seed

        # if there is none (e.g. when the batch is created manually), make a random seed
        self._local.random_seed = np.random.SeedSequence()
        return self._local.random_seed

    @random_seed.setter
    def random_seed(self, value):
        """ : SeedSequence for random number generation """
        self._local.random_seed = value
        self._local.random = make_rng(value)

    def __copy__(self):
        dump_batch = dill.dumps(self)
        restored_batch = dill.loads(dump_batch)
        return restored_batch

    def deepcopy(self):
        """ Return a deep copy of the batch. """
        return self.__copy__()

    @classmethod
    def from_data(cls, index=None, data=None):
        """ Create a batch from data given """
        # this is roughly equivalent to self.data = data
        if index is None:
            index = np.arange(len(data))
        return cls(index, preloaded=data)

    @classmethod
    def merge(cls, batches, batch_size=None, components=None, batch_class=None):
        """ Merge several batches to form a new batch of a given size

        Parameters
        ----------
        batches : tuple of batches

        batch_size : int or None
            if `None`, just merge all batches into one batch (the rest will be `None`),
            if `int`, then make one batch of `batch_size` and a batch with the rest of data.

        components : str, tuple or None
            if `None`, all components from initial batches will be created,
            if `str` or `tuple`, then create these components in new batches.

        batch_class : Batch or None
            if `None`, created batches will be of the same class as initial batch,
            if `Batch`, created batches will be of that class.

        Returns
        -------
        batch, rest : tuple of two batches

        Raises
        ------
        ValueError
            If component is `None` in some batches and not `None` in others.
        """
        batch_class = batch_class or cls
        def _make_index(data):
            return DatasetIndex(data.shape[0]) if data is not None and data.shape[0] > 0 else None

        def _make_batch(data):
            index = _make_index(data[0])
            batch = batch_class.from_data(index, tuple(data)) if index is not None else None
            if batch is not None:
                batch.components = tuple(components)
                _ = batch.data
            return batch

        if batch_size is not None:
            break_point = len(batches) - 1
            last_batch_len = 0
            cur_size = 0
            for i, b in enumerate(batches):
                cur_batch_len = len(b)
                if cur_size + cur_batch_len >= batch_size:
                    break_point = i
                    last_batch_len = batch_size - cur_size
                    break

                cur_size += cur_batch_len
                last_batch_len = cur_batch_len

        if components is None:
            components = batches[0].components or (None,)
        elif isinstance(components, str):
            components = (components, )
        new_data = list(None for _ in components)
        rest_data = list(None for _ in components)

        for i, comp in enumerate(components):
            none_components_in_batches = [b.get(component=comp) is None for b in batches]
            if np.all(none_components_in_batches):
                continue
            if np.any(none_components_in_batches):
                raise ValueError(f'Component {comp} is None in some batches')

            if batch_size is None:
                new_comp = [b.get(component=comp) for b in batches]
            else:
                last_batch = batches[break_point]
                new_comp = [b.get(component=comp) for b in batches[:break_point]] + \
                           [last_batch.get(component=comp)[:last_batch_len]]

            new_data[i] = cls.merge_component(comp, new_comp)

            if batch_size is not None:
                rest_comp = [last_batch.get(component=comp)[last_batch_len:]] + \
                            [b.get(component=comp) for b in batches[break_point + 1:]]
                rest_data[i] = cls.merge_component(comp, rest_comp)

        new_batch = _make_batch(new_data)
        rest_batch = _make_batch(rest_data)

        return new_batch, rest_batch

    @classmethod
    def merge_component(cls, component=None, data=None):
        """ Merge the same component data from several batches """
        _ = component
        if isinstance(data[0], np.ndarray):
            return np.concatenate(data)
        raise TypeError("Unknown data type", type(data[0]))

    def as_dataset(self, dataset=None, copy=False):
        """ Makes a new dataset from batch data

        Parameters
        ----------
        dataset
            an instance or a subclass of Dataset

        copy : bool
            whether to copy batch data to allow for further inplace transformations

        Returns
        -------
        an instance of a class specified by `dataset` arg, preloaded with this batch data
        """
        dataset = dataset or self._dataset
        if dataset is None:
            raise ValueError('dataset can be an instance of Dataset (sub)class or the class itself, but not None')
        if isinstance(dataset, type):
            dataset_class = dataset
            attrs = {}
        else:
            dataset_class = dataset.__class__
            attrs = dataset.get_attrs()
        return dataset_class(self.index, batch_class=type(self), preloaded=self._data, copy=copy, **attrs)

    @property
    def indices(self):
        """: numpy array - an array with the indices """
        if isinstance(self.index, DatasetIndex):
            return self.index.indices
        return self.index

    def __len__(self):
        return len(self.index)

    @property
    def size(self):
        """: int - number of items in the batch """
        return len(self)

    @action
    def add_components(self, components, init=None):
        """ Add new components

        Parameters
        ----------
        components : str or list
            new component names
        init : array-like
            initial component data

        Raises
        ------
        ValueError
            If a component or an attribute with the given name already exists
        """
        if isinstance(components, str):
            components = (components,)
            init = (init,)
        elif isinstance(components, (tuple, list)):
            components = tuple(components)
            if init is None:
                init = (None,) * len(components)
            else:
                init = tuple(init)

        for comp, value in zip(components, init):
            if hasattr(self, comp):
                raise ValueError(f"An attribute '{comp}' already exists")
            if self.components is not None and comp in self.components:
                raise ValueError(f"A components '{comp}' already exists")

            if self.components is None:
                self.components = tuple([comp])
                if self._data is not None:
                    warnings.warn("All batch data is erased")
            else:
                self.components = self.components + tuple([comp])
            setattr(self, comp, value)

        return self

    def __getattr__(self, name):
        if self.components is not None and name in self.components:   # pylint: disable=unsupported-membership-test
            return getattr(self.data, name, None)
        raise AttributeError(f"{name} not found in class {self.__class__.__name__}")

    def __setattr__(self, name, value):
        if self.components is not None:
            if name == "_data":
                super().__setattr__(name, value)
                if self._data is not None:
                    if isinstance(self._data, BaseComponents):
                        self._data_named = self._data
                    else:
                        self._data_named = create_item_class(self.components, self._data)
                return
            if name in self.components:    # pylint: disable=unsupported-membership-test
                # preload data if needed
                _ = self.data
                if self._data_named is None or self._data_named.components != self.components:
                    self._data_named = create_item_class(self.components, self._data)
                setattr(self._data_named, name, value)
                # update _data with with new component values
                super().__setattr__('_data', self._data_named.data)
                return
        super().__setattr__(name, value)

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_local'] = state['_local'] is not None
        state['_preloaded_lock'] = True
        return state

    def __setstate__(self, state):
        state['_preloaded_lock'] = threading.Lock() if state['_preloaded_lock'] else None
        state['_local'] = threading.local() if state['_local'] else None

        for k, v in state.items():
            # this warrants that all hidden objects are reconstructed upon unpickling
            setattr(self, k, v)

    @property
    def array_of_nones(self):
        """1-D ndarray: ``NumPy`` array with ``None`` values."""
        return np.array([None] * len(self.index))

    def get(self, item=None, component=None):
        """ Return an item from the batch or the component """
        if item is None:
            if component is None:
                res = self.data
            else:
                res = getattr(self, component)
        else:
            if component is None:
                res = self[item]
            else:
                res = getattr(self[item], component)
        return res

    def __getitem__(self, item):
        return self.data[item] if self.data is not None else None

    def __iter__(self):
        for item in self.indices:
            yield self[item]

    @property
    def items(self):
        """: list - batch items """
        return [[self[ix]] for ix in self.indices]

    def run_once(self, *args, **kwargs):
        """ Init function for no parallelism
        Useful for async action-methods (will wait till the method finishes)
        """
        _ = self.data, args, kwargs
        return [[]]

    def get_errors(self, all_res):
        """ Return a list of errors from a parallel action """
        all_errors = [error for error in all_res if isinstance(error, Exception)]
        return all_errors if len(all_errors) > 0 else None

    @action
    def do_nothing(self, *args, **kwargs):
        """ An empty action (might be convenient in complicated pipelines) """
        _ = args, kwargs
        return self


    @action
    def apply_parallel(self, func, init=None, post=None, src=None, dst=None, *args,
                       p=None, target='for', requires_rng=False, rng_seeds=None, **kwargs):
        """ Apply a function to each item in the container, returned by `init`, and assemble results by `post`.
        Depending on the `target` parameter, different parallelization engines may be used: for, threads, MPC, async.

        Roughly, under the hood we perform the following:
            - compute parameters, individual for each worker. Currently, these are:
                - `p` to indicate whether the function should be applied
                - worker id and a seed for random generator, if required
            - call `init` function, which outputs a container of items, passed directly to the `func`.
            The simplest example is the `init` funciton that returns batch indices, and the function works off of each.
            - wrap the `func` call into parallelization engine of choice.
            - compute results of `func` calls for each item, returned by `init`
            - assemble results by `post` function, e.g. stack the obtained numpy arrays.

        In the simplest possible case of `init=None`, `src=images`, `dst=images_transformed`, `post=None`,
        this function is almost equivalent to:
            container = [func(item, *args, **kwargs) for item in self.images]
            self.images_transformed = container

        If `src` is a list and `dst` is a list, then this function is applied recursively to each pair of src, dst.
        If `src` is a tuple, then this tuple is used as a whole.
        This allows to make functions that work on multiple components.

        Parameters
        ----------
        func : callable
            A function to apply to each item from the source.
            Should accept `src` and `dst` parameters, or be written in a way that accepts variable args.
        target : str
            Parallelization engine:
                - 'f', 'for' for executing each worker sequentially, like in a for-loop.
                - 't', 'threads' for using threads.
                - 'm', 'mpc' for using processes. Note the bigger overhead for process initialization.
                - 'a', 'async' for asynchronous execution.
        init : str, callable or container
            Function to init data for individual workers: must return a container of items.

            If 'data', then use `src` components as the init.
            If other str, then must be a name of the attribute of the batch to use as the init.
            If callable or any previous returned a callable, then result of this callable is used as the init.
            Note that in the last case callable should accept `src` and `dst` parameters, and `kwargs` are also passed.
            If not any of the above, then the object is used directly, for example, np.ndarray.

        post : str or callable
            Function to apply to the results of function evaluation on each item.
            Must accept `src` and `dst` parameters, as well as `kwargs`.

        src : str, sequence, list of str
            The source to get data from:
            - None
            - str - a component name, e.g. 'images' or 'masks'
            - tuple or list of str - several component names
            - sequence - data as a numpy-array, data frame, etc

        dst : str or array
            the destination to put the result in, can be:
            - None - in this case dst is set to be same as src
            - str - a component name, e.g. 'images' or 'masks'
            - tuple or list of str, e.g. ['images', 'masks']
            If not provided, uses `src`.

        p : float or None
            Probability of applying func to an element in the batch.

        requires_rng : bool
            Whether the `func` requires RNG. Should be used for correctly initialized seeds for reproducibility.
            If True, then a pre-initialized RNG will be passed to the function call as `rng` keyword parameter.

        args, kwargs
            Other parameters passed to ``func``.

        Notes
        -----
        apply_parallel does the following (but in parallel)::

            for item in range(len(batch)):
                self.dst[item] = func(self.src[item], *args, **kwargs)

        `apply_parallel(func, src=['images', 'masks'])` is equal to
        `apply_parallel(func, src=['images', 'masks'], dst=['images', 'masks'])`,
        which in turn equals to two subsequent calls::

            images = func(images)
            masks = func(masks)

        However, named expressions will be evaluated only once before the first call.

        Whereas `apply_parallel(func, src=('images', 'masks'))` (i.e. when `src` takes a tuple of component names,
        not the list as in the previous example) passes both components data into `func` simultaneously::

            images, masks = func((images, masks))

        Examples
        --------
        ::

            apply_parallel(make_masks_fn, src='images', dst='masks')
            apply_parallel(apply_mask, src=('images', 'masks'), dst='images_with_masks')
            apply_parallel(rotate, src=['images', 'masks'], dst=['images', 'masks'], p=.2)
            apply_parallel(MyBatch.some_static_method, p=.5)
            apply_parallel(B.some_method, src='features', p=.5)

        TODO: move logic of applying `post` function from `inbatch_parallel` here, as well as remove `use_self` arg.
        """
        #pylint: disable=keyword-arg-before-vararg
        # Parse parameters: fill with class-wide defaults
        init = init or self.apply_defaults.get('init', None)
        post = post or self.apply_defaults.get('post', None)
        target = target or self.apply_defaults.get('target', None)

        # Prepare parameters, individual for each worker: probability of applying, RNG seed, id
        if isinstance(p, float):
            p = P(R('binomial', 1, p, seed=self.random)).get(batch=self)

        if requires_rng and rng_seeds is None:
            rng_seeds = P(R('integers', 0, 9223372036854775807, seed=self.random)).get(batch=self)

        worker_ids = P(np.arange(len(self), dtype=np.int32))

        # Case of list `src`: recursively call for each pair of src/dst
        if isinstance(src, list) and not (dst is None or isinstance(dst, list) and len(src) == len(dst)):
            raise ValueError("src and dst must have equal length")
        if isinstance(src, list) and (dst is None or isinstance(dst, list) and len(src) == len(dst)):
            if dst is None:
                dst = src

            for src_, dst_ in zip(src, dst):
                self.apply_parallel(func=func, init=init, post=post, src=src_, dst=dst_,
                                    *args, p=p, target=target, rng_seeds=rng_seeds, **kwargs)
            return self

        # Actual computation
        if init is None or init is False or init == 'data':
            if isinstance(src, str):
                init = self.get(component=src)
            elif isinstance(src, (tuple, list)):
                init = list((x,) for x in zip(*[self.get(component=s) for s in src]))
            else:
                init = src
        elif isinstance(init, str):
            # No hasattr check: if it is False, then an error would (and should) be raised
            init = getattr(self, init)
            if callable(init):
                init = init(src=src, dst=dst, p=p, target=target, **kwargs)

        # Compute result. Unbind the method to pass self explicitly
        parallel = inbatch_parallel(init=init, post=post, target=target, src=src, dst=dst)
        transform = parallel(type(self)._apply_once)
        result = transform(self, *args, func=func, p=p, src=src, dst=dst,
                           apply_parallel_id=worker_ids, apply_parallel_seeds=rng_seeds, **kwargs)
        return result

    def _apply_once(self, item, *args, func=None, p=None, apply_parallel_id=None, apply_parallel_seeds=None, **kwargs):
        """ Apply a function to each item in the batch.

        Parameters
        ----------
        item
            An item from `init` function.
        func : callable
            A function to apply to each item.
        p : None or {0, 1}
            Whether to apply func to an element in the batch. If not specified, counts as 1.
            Created and distributed to individual items by :meth:``.apply_parallel`.
        apply_parallel_id : None, int
            Index of the current item in the overall `init`.
            Created and distributed to individual items by :meth:``.apply_parallel`.
        apply_parallel_seeds : None, int
            If provided, then the seed to create RNG for this given worker.
            If provided, then supplied to a function call as `rng` keyword parameter.
            Created and distributed to individual items by :meth:``.apply_parallel`.
        args, kwargs
            Other parameters passed to ``func``.
        """
        _ = apply_parallel_id

        if p is None or p == 1:
            if apply_parallel_seeds is not None:
                rng = np.random.default_rng(np.random.SFC64(apply_parallel_seeds))
                kwargs['rng'] = rng
            return func(item, *args, **kwargs)
        return item

    def _get_file_name(self, ix, src):
        """ Get full path file name corresponding to the current index.

        Parameters
        ----------
        src : str, FilesIndex or None
            if None, full path to the indexed item will be returned.
            if FilesIndex it must contain the same indices values as in the self.index.
            if str, behavior depends on wheter self.index.dirs is True. If self.index.dirs is True
            then src will be appended to the end of the full paths from self.index. Else if
            self.index.dirs is False then src is considered as a directory name and the basenames
            from self.index will be appended to the end of src.

        Examples
        --------
        Let folder "/some/path/*.dcm" contain files "001.png", "002.png", etc. Then if self.index
        was built as

        >>> index = FilesIndex(path="/some/path/*.png", no_ext=True)

        Then _get_file_name(ix, src="augmented_images/") will return filenames:
        "augmented_images/001.png", "augmented_images/002.png", etc.


        Let folder "/some/path/*" contain folders "001", "002", etc. Then if self.index
        was built as

        >>> index = FilesIndex(path="/some/path/*", dirs=True)

        Then _get_file_name(ix, src="masks.png") will return filenames:
        "/some/path/001/masks.png", "/some/path/002/masks.png", etc.


        If you have two directories "images/*.png", "labels/*png" with identical filenames,
        you can build two instances of FilesIndex and use the first one to biuld your Dataset

        >>> index_images = FilesIndex(path="/images/*.png", no_ext=True)
        >>> index_labels = FilesIndex(path="/labels/*.png", no_ext=True)
        >>> dset = Dataset(index=index_images, batch_class=Batch)

        Then build dataset using the first one
        _get_file_name(ix, src=index_labels) to reach corresponding files in the second path.

        """
        if not isinstance(self.index, FilesIndex):
            raise ValueError("File locations must be specified to dump/load data")

        if isinstance(src, str):
            if self.index.dirs:
                fullpath = self.index.get_fullpath(ix)
                file_name = os.path.join(fullpath, src)

            else:
                file_name = os.path.basename(self.index.get_fullpath(ix))
                file_name = os.path.join(os.path.abspath(src), file_name)

        elif isinstance(src, FilesIndex):
            try:
                file_name = src.get_fullpath(ix)
            except KeyError as e:
                raise KeyError(f"File {ix} is not indexed in the received index") from e

        elif src is None:
            file_name = self.index.get_fullpath(ix)

        else:
            raise ValueError("Src must be either str, FilesIndex or None")

        return file_name

    def _assemble_component(self, result, *args, component, **kwargs):
        """ Assemble one component after parallel execution.

        Parameters
        ----------
        result : sequence, np.ndarray
            Values to put into ``component``
        component : str
            Component to assemble.
        """

        _ = args, kwargs
        try:
            new_items = np.stack(result)
        except ValueError as e:
            message = str(e)
            if "must have the same shape" in message:
                new_items = np.empty(len(result), dtype=object)
                new_items[:] = result
            else:
                raise e

        if hasattr(self, component):
            setattr(self, component, new_items)
        else:
            self.add_components(component, new_items)

    def _assemble(self, all_results, *args, dst=None, **kwargs):
        """ Assembles the batch after a parallel action.

        Parameters
        ----------
        all_results : sequence
            Results after inbatch_parallel.
        dst : str, sequence, np.ndarray
            Components to assemble

        Returns
        -------
        self
        """

        _ = args
        if any_action_failed(all_results):
            all_errors = self.get_errors(all_results)
            print(all_errors[0])
            traceback.print_tb(all_errors[0].__traceback__)
            raise RuntimeError("Could not assemble the batch") from all_errors[0]

        if dst is None:
            dst_default = kwargs.get('dst_default', 'src')
            if dst_default == 'src':
                dst = kwargs.get('src')
            elif dst_default == 'components':
                dst = self.components

        if not isinstance(dst, (list, tuple, np.ndarray)):
            dst = [dst]

        if len(dst) == 1:
            all_results = [all_results]
        else:
            all_results = list(zip(*all_results))

        for component, result in zip(dst, all_results):
            self._assemble_component(result, component=component, **kwargs)
        return self

    @apply_parallel_(init='indices', post='_assemble', target='f', dst_default='components')
    def _load_blosc(self, ix, src=None, dst=None):
        """ Load data from a blosc packed file """
        file_name = self._get_file_name(ix, src)
        with open(file_name, 'rb') as f:
            data = dill.loads(blosc.decompress(f.read()))
            components = tuple(dst or self.components)
            try:
                item = tuple(data[i] for i in components)
            except Exception as e:
                raise KeyError('Cannot find components in corresponfig file', file_name) from e
        return item

    @apply_parallel_(init='indices', target='f')
    def _dump_blosc(self, ix, dst=None, components=None):
        """ Save blosc packed data to file """
        file_name = self._get_file_name(ix, dst)
        with open(file_name, 'w+b') as f:
            if self.components is None:
                components = (None,)
                item = (self[ix],)
            else:
                components = tuple(components or self.components)
                item = self[ix].as_tuple(components)
            data = dict(zip(components, item))
            f.write(blosc.compress(dill.dumps(data)))

    def _load_table(self, src, fmt, dst=None, post=None, *args, **kwargs):
        """ Load a data frame from table formats: csv, hdf5, feather """
        if fmt == 'csv':
            _data = pd.read_csv(src, *args, **kwargs)
        elif fmt == 'feather':
            _data = feather.read_dataframe(src, *args, **kwargs)
        elif fmt == 'hdf5':
            _data = pd.read_hdf(src, *args, **kwargs)

        # Put into this batch only part of it (defined by index)
        if isinstance(_data, pd.DataFrame):
            _data = _data.loc[self.indices]
        elif isinstance(_data, dd.DataFrame):
            # dask.DataFrame.loc supports advanced indexing only with lists
            _data = _data.loc[list(self.indices)].compute()

        if callable(post):
            _data = post(_data, src=src, fmt=fmt, dst=dst, **kwargs)

        self.load(src=_data, dst=dst)


    @action(use_lock='__dump_table_lock')
    def _dump_table(self, dst, fmt='feather', components=None, *args, **kwargs):
        """ Save batch data to table formats

        Args:
          dst: str - a path to dump into
          fmt: str - format: feather, hdf5, csv
          components: str or tuple - one or several component names
        """
        filename = dst

        components = tuple(components or self.components)
        data_dict = {}
        for comp in components:
            comp_data = self.get(component=comp)
            if isinstance(comp_data, pd.DataFrame):
                data_dict.update(comp_data.to_dict('series'))
            elif isinstance(comp_data, np.ndarray):
                if comp_data.ndim > 1:
                    columns = [comp + str(i) for i in range(comp_data.shape[1])]
                    comp_dict = zip(columns, (comp_data[:, i] for i in range(comp_data.shape[1])))
                    data_dict.update({comp: comp_dict})
                else:
                    data_dict.update({comp: comp_data})
            else:
                data_dict.update({comp: comp_data})
        _data = pd.DataFrame(data_dict)

        if fmt == 'feather':
            feather.write_dataframe(_data, filename, *args, **kwargs)
        elif fmt == 'hdf5':
            _data.to_hdf(filename, *args, **kwargs)   # pylint:disable=no-member
        elif fmt == 'csv':
            _data.to_csv(filename, *args, **kwargs)   # pylint:disable=no-member
        else:
            raise ValueError(f'Unknown format {fmt}')

        return self

    def _load_from_source(self, dst, src):
        """ Load data from a memory object (tuple, ndarray, pd.DataFrame, etc) """
        if dst is None:
            self._data = create_item_class(self.components, source=src, indices=self.indices,
                                           crop=True, copy=self._copy)
        else:
            if isinstance(dst, str):
                dst = (dst,)
                src = (src,)
            source = create_item_class(dst, source=src, indices=self.indices, crop=True, copy=self._copy)
            for comp in dst:
                setattr(self, comp, getattr(source, comp))

    @action
    def load(self, *args, src=None, fmt=None, dst=None, **kwargs):
        """ Load data from another array or a file.

        Parameters
        ----------
        src :
            a source (e.g. an array or a file name)

        fmt : str
            a source format, one of None, 'blosc', 'csv', 'hdf5', 'feather'

        dst : None or str or tuple of str
            components to load `src` to

        **kwargs :
            other parameters to pass to format-specific loaders

        Notes
        -----
        Loading creates new components if necessary.

        Examples
        --------
        Load data from a pandas dataframe's columns into all batch components::

            batch.load(src=dataframe)

        Load data from dataframe's columns `features` and `labels` into components `features` and `labels`::

            batch.load(src=dataframe, dst=('features', 'labels'))

        Load a dataframe into a component `features`::

            batch.load(src=dataframe, dst='features')

        Load data from a dict into components `images` and `masks`::

            batch.load(src=dict(images=images_array, masks=masks_array), dst=('images', 'masks'))

        Load data from a tuple into components `images` and `masks`::

            batch.load(src=(images_array, masks_array), dst=('images', 'masks'))

        Load data from an array into a component `images`::

            batch.load(src=images_array, dst='images')

        Load data from a CSV file columns into components `features` and `labels`::

            batch.load(fmt='csv', src='/path/to/file.csv', dst=('features', 'labels`), index_col=0)
        """
        _ = args

        if dst is not None:
            self.add_components(np.setdiff1d(dst, self.components).tolist())

        if fmt is None:
            self._load_from_source(src=src, dst=dst)
        elif fmt == 'blosc':
            self._load_blosc(src=src, dst=dst, **kwargs)
        elif fmt in ['csv', 'hdf5', 'feather']:
            self._load_table(src=src, fmt=fmt, dst=dst, **kwargs)
        else:
            raise ValueError("Unknown format " + fmt)
        return self

    @action
    def dump(self, *args, dst=None, fmt=None, components=None, **kwargs):
        """ Save data to another array or a file.

        Parameters
        ----------
        dst :
            a destination (e.g. an array or a file name)

        fmt : str
            a destination format, one of None, 'blosc', 'csv', 'hdf5', 'feather'

        components : None or str or tuple of str
            components to load

        *args :
            other parameters are passed to format-specific writers

        *kwargs :
            other parameters are passed to format-specific writers
        """
        components = [components] if isinstance(components, str) else components
        if fmt is None:
            if components is not None and len(components) > 1:
                raise ValueError("Only one component can be dumped into a memory array: components =", components)
            components = components[0] if components is not None else None
            dst[self.indices] = self.get(component=components)
        elif fmt == 'blosc':
            self._dump_blosc(dst, components=components)
        elif fmt in ['csv', 'hdf5', 'feather']:
            self._dump_table(dst, fmt, components, *args, **kwargs)
        else:
            raise ValueError("Unknown format " + fmt)
        return self

    @action
    def save(self, *args, **kwargs):
        """ Save batch data to a file (an alias for dump method)"""
        return self.dump(*args, **kwargs)

    @apply_parallel_
    def to_array(self, comp, dtype=np.float32, channels='last', **kwargs):
        """ Converts batch components to np.ndarray format

        Parameters
        ----------
        src : str
            Component to get images from. Default is 'images'.
        dst : str
            Component to write images to. Default is 'images'.
        dtype : str or np.dtype
            Data type
        channels : None, 'first' or 'last'
            the dimension for channels axis
        """
        _ = kwargs
        comp = np.array(comp)

        if len(comp.shape) == 2:
            # a special treatment for 2d arrays with images - add a new dimension for channels
            if channels == 'first':
                comp = comp[np.newaxis, :, :]
            elif channels == 'last':
                comp = comp[:, :, np.newaxis]
        else:
            # we assume that channels is 'last' by default
            # so move channels from the last to the first axis if needed
            if channels == 'first':
                comp = np.moveaxis(comp, -1, 0)

        if dtype is not None:
            comp = comp.astype(dtype)

        return comp
