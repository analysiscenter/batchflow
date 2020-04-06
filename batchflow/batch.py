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
    import _fake as pd
try:
    import feather
except ImportError:
    pass
try:
    import dask.dataframe as dd
except ImportError:
    import _fake as dd

from .dsindex import DatasetIndex, FilesIndex
from .decorators import action, inbatch_parallel, any_action_failed
from .components import create_item_class, BaseComponents


class MethodsTransformingMeta(type):
    """ A metaclass to transform all class methods in the way described below:

        1. Wrap method with either `apply_transform` or `apply_transform_all`
           depending on the value of `all` argument (from `transform_kwargs`),
           which is set via decorator @apply_transform. Then add this
           wrapped method to a class namespace by its original name.

        2. Add the original version of the method (i.e. unwrapped) to a class
           namespace using name with underscores: `'_{}_'.format(name)`. This
           is necessary in order to allow inner calls of untransformed versions
           (e.g. `ImagesBatch.scale` calls `ImagesBatch.crop` under the hood).
    """
    def __new__(cls, name, bases, namespace):
        namespace_ = namespace.copy()
        for object_name, object_ in namespace.items():
            transform_kwargs = getattr(object_, 'transform_kwargs', None)
            if transform_kwargs is not None:
                namespace_[object_name] = cls.apply_transform(object_, **transform_kwargs)

                disclaimer = "This is an untransformed version of `{}`.\n\n".format(object_.__qualname__)
                object_.__doc__ = disclaimer + object_.__doc__
                object_.__name__ = '_' + object_name + '_'
                object_.__qualname__ = '.'.join(object_.__qualname__.split('.')[:-1] + [object_.__name__])
                namespace_[object_.__name__] = object_

        return super().__new__(cls, name, bases, namespace_)

    @classmethod
    def apply_transform(cls, method, **transform_kwargs):
        """ Wrap passed `method` in accordance with `all` arg value """
        @functools.wraps(method)
        def apply_transform_wrapper(self, *args, **kwargs):
            transform = self.apply_transform
            method_ = method.__get__(self, type(self)) # bound method to class
            transform_kwargs_full = {**self.transform_defaults, **transform_kwargs}
            all = transform_kwargs_full.pop('all')
            if all:
                transform = self.apply_transform_all
                _ = [transform_kwargs_full.pop(keyname) for keyname in ['target', 'init', 'post']]
            kwargs_full = {**transform_kwargs_full, **kwargs}
            return transform(method_, *args, **kwargs_full)
        return action(apply_transform_wrapper)


class Batch(metaclass=MethodsTransformingMeta):
    """ The core Batch class

    Note, that if any class method is wrapped with `@apply_transform` decorator
    than for inner calls (i.e. from other class methods) should be used version
    of desired method with underscores. (For example, if there is a decorated
    `method` than you need to call `_method_` from inside of `other_method`).
    Same is applicable for all child classes of :class:`batch.Batch`.
    """
    components = None
    # Class-specific defaults for :meth:`.Batch.apply_transform`
    transform_defaults = dict(target='threads',
                              init='indices',
                              post='_assemble',
                              src=None,
                              dst=None,
                              all=False)

    def __init__(self, index, dataset=None, pipeline=None, preloaded=None, copy=False, *args, **kwargs):
        _ = args
        if  self.components is not None and not isinstance(self.components, tuple):
            raise TypeError("components should be a tuple of strings with components names")
        self.index = index
        self._preloaded_lock = threading.Lock()
        self._preloaded = preloaded
        self._copy = copy
        self._local = None
        self._data_named = None
        self._data = None
        self._dataset = dataset
        self._pipeline = pipeline
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
        if self._data is None and self._preloaded is not None:
            # load data the first time it's requested
            with self._preloaded_lock:
                if self._data is None and self._preloaded is not None:
                    self.load(src=self._preloaded)
        res = self._data if self.components is None else self._data_named
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
        if self._local is not None and hasattr(self._local, 'pipeline'):
            return self._local.pipeline
        return self._pipeline

    @pipeline.setter
    def pipeline(self, val):
        """ Store pipeline in a thread-local storage """
        if val is None:
            self._local = None
        else:
            if self._local is None:
                self._local = threading.local()
            self._local.pipeline = val
        self._pipeline = val

    def __copy__(self):
        pipeline = self.pipeline
        self.pipeline = None
        dump_batch = dill.dumps(self)
        self.pipeline = pipeline

        restored_batch = dill.loads(dump_batch)
        restored_batch.pipeline = pipeline
        return restored_batch

    def deepcopy(self):
        """ Return a deep copy of the batch.

        Constructs a new ``Batch`` instance and then recursively copies all
        the objects found in the original batch, except the ``pipeline``,
        which remains unchanged.

        Returns
        -------
        Batch
        """
        return self.copy()

    @classmethod
    def from_data(cls, index, data):
        """ Create a batch from data given """
        # this is roughly equivalent to self.data = data
        if index is None:
            index = np.arange(len(data))
        return cls(index, preloaded=data)

    @classmethod
    def from_batch(cls, batch):
        """ Create batch from another batch """
        return cls(batch.index, dataset=batch.dataset, pipeline=batch.pipeline, preloaded=batch.data,
                   **batch.get_attrs())

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
            if `str` or `tuple`, then create thay components in new batches.
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

        if batch_size is None:
            break_point = len(batches)
        else:
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
                raise ValueError('Component {} is None in some batches'.format(comp))

            if batch_size is None:
                new_comp = [b.get(component=comp) for b in batches[:break_point]]
            else:
                last_batch = batches[break_point]
                last_batch_last_index = last_batch.get_pos(None, comp, last_batch.indices[last_batch_len - 1])
                new_comp = [b.get(component=comp) for b in batches[:break_point]] + \
                           [last_batch.get(component=comp)[:last_batch_last_index + 1]]
            new_data[i] = cls.merge_component(comp, new_comp)

            if batch_size is not None:
                rest_comp = [last_batch.get(component=comp)[last_batch_last_index + 1:]] + \
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
                raise ValueError("An attribute '%s' already exists" % comp)
            if self.components is not None and comp in self.components:
                raise ValueError("A components '%s' already exists" % comp)

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
            attr = getattr(self.data, name, None)
            return attr
        raise AttributeError("%s not found in class %s" % (name, self.__class__.__name__))

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
        state.pop('_data_named')
        state['_local'] = state['_local'] is not None
        state['_preloaded_lock'] = True
        return state

    def __setstate__(self, state):
        state['_preloaded_lock'] = threading.Lock() if state['_preloaded_lock'] else None
        state['_local'] = threading.Lock() if state['_local'] else None

        for k, v in state.items():
            # this warrants that all hidden objects are reconstructed upon unpickling
            setattr(self, k, v)

    @property
    def array_of_nones(self):
        """1-D ndarray: ``NumPy`` array with ``None`` values."""
        return np.array([None] * len(self.index))

    def get_pos(self, data, component, index):
        """ Return a position in data for a given index

        Parameters
        ----------
        data : some array or tuple of arrays
            if `None`, should return a position in :attr:`self.data <.Batch.data>`

        components : None, int or str
            - None - data has no components (e.g. just an array or pandas.DataFrame)
            - int - a position of a data component, when components names are not defined
                (e.g. data is a tuple)
            - str - a name of a data component

        index : any
            an index id

        Returns
        -------
        int
            a position in a batch data where an item with a given index is stored

        Notes
        -----
        It is used to read / write data from / to a given component::

            batch_data = data.component[pos]
            data.component[pos] = new_data

        if `self.data` holds a numpy array, then get_pos(None, None, index) should
        just return `self.index.get_pos(index)`

        if `self.data.images` contains BATCH_SIZE images as a numpy array,
        then `get_pos(None, 'images', index)` should return `self.index.get_pos(index)`

        if `self.data.labels` is a dict {index: label}, then `get_pos(None, 'labels', index)` should return index.

        if `data` is not `None`, then you need to know in advance how to get a position for a given index.

        For instance, `data` is a large numpy array, and a batch is a subset of this array and
        `batch.index` holds row numbers from a large arrays.
        Thus, `get_pos(data, None, index)` should just return index.

        A more complicated example of data:

        - batch represent small crops of large images
        - `self.data.source` holds a few large images (e.g just 5 items)
        - `self.data.coords` holds coordinates for crops (e.g. 100 items)
        - `self.data.image_no` holds an array of image numbers for each crop (so it also contains 100 items)

        then `get_pos(None, 'source', index)` should return `self.data.image_no[self.index.get_pos(index)]`.
        Whilst, `get_pos(data, 'source', index)` should return `data.image_no[index]`.
        """
        _ = component
        if data is None: # or data is self._data:
            pos = self.index.get_pos(index)
        else:
            pos = index
        return pos

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

    def get_model_by_name(self, model_name):
        """ Return a model specification given its name """
        return self.pipeline.get_model_by_name(model_name, batch=self)

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
    def apply_transform(self, func, *args, **kwargs):
        """ Apply a function to each item in the batch.

        Notes
        -----
        Redefine :attr:`self.transform_defaults <.Batch.transform_defaults>` in
        child classes. This is proposed solely for the purposes of brevity — in
        order to avoid repeated heavily loaded class methods decoration, e.g.
        `@apply_transform(init='indices', target='for', src='images')` which in
        most cases is actually equivalent to simple `@apply_transform` assuming
        that the defaults are redefined for the class whose methods are being
        transformed. Note, that if no defaults redefined those from the nearest
        parent class will be used in :class:`batch.MethodsTransformingMeta`.

        Parameters
        ----------
        func : callable
            a function to apply to each item from the source

        target : str
            See :func:`~batchflow.inbatch_parallel` for details.

        init : str, callable or iterable
            See :func:`~batchflow.inbatch_parallel` for details.

        post : str or callable
            See :func:`~batchflow.inbatch_parallel` for details.

        src : str, sequence, list of str
            the source to get data from, can be:
            - None
            - str - a component name, e.g. 'images' or 'masks'
            - sequence - a numpy-array, list, etc
            - tuple of str - get data from several components
            - list of str, sequences or tuples - apply same transform to each item in list

        dst : str or array
            the destination to put the result in, can be:
            - None - in this case dst is set to be same as src
            - str - a component name, e.g. 'images' or 'masks'
            - tuple of list of str, e.g. ['images', 'masks']
            if src is a list, dst should be either list or None.

        p : float or None
            probability of applying transform to an element in the batch

            if not None, indices of relevant batch elements will be passed ``func``
            as a named arg ``indices``.

        args, kwargs
            other parameters passed to ``func``

        Notes
        -----
        apply_transform does the following (but in parallel)::

            for item in range(len(batch)):
                self.dst[item] = func(self.src[item], *args, **kwargs)

        If `src` is a list with two or more elements, `dst` should be list or
        tuple of the same lenght.

        Examples
        --------

        ::

            apply_transform(make_masks_fn, src='images', dst='masks')
            apply_transform(apply_mask, src=('images', 'masks'), dst='images')
            FIXME apply_transform(rotate, src=['images', 'masks'], dst=['images', 'masks'], p=.2)
            apply_transform(MyBatch.some_static_method, p=.5)
            apply_transform(B.some_method, p=.5)
        """
        kwargs_full = {**self.transform_defaults, **kwargs}
        target, init, post, _ = [kwargs_full.pop(keyname) for keyname in ['target', 'init', 'post', 'all']]

        parallel = inbatch_parallel(init=init, post=post, target=target)
        transform = parallel(type(self)._apply_transform)
        return transform(self, func, *args, **kwargs_full)

    def _apply_transform(self, ix, func, *args, src=None, dst=None, p=None, **kwargs):
        """ Apply a function to each item in the batch.

        Parameters
        ----------
        func : callable
            a function to apply to each item from the source

        src : str, sequence, list of str
            the source to get data from, can be:
            - None
            - str - a component name, e.g. 'images' or 'masks'
            - sequence - a numpy-array, list, etc
            - tuple of str - get data from several components

        dst : str or array
            the destination to put the result in, can be:
            - None
            - str - a component name, e.g. 'images' or 'masks'
            - array-like - a numpy-array, list, etc

        p : float or None
            probability of applying transform to an element in the batch

            if not None, indices of relevant batch elements will be passed ``func``
            as a named arg ``indices``.

        args, kwargs
            other parameters passed to ``func``
        """
        dst = src if dst is None else dst

        if not (isinstance(dst, str) or
                (isinstance(dst, (list, tuple)) and np.all([isinstance(component, str) for component in dst]))):
            raise TypeError("dst should be str or tuple or list of str")

        if src is None:
            _args = args
        else:
            if isinstance(src, str):
                pos = self.get_pos(None, src, ix)
                src_attr = (getattr(self, src)[pos],)
            elif isinstance(src, (tuple, list)) and np.all([isinstance(component, str) for component in src]):
                src_attr = [getattr(self, component)[self.get_pos(None, component, ix)] for component in src]
            else:
                pos = self.get_pos(None, dst, ix)
                src_attr = (src[pos],)
            _args = tuple([*src_attr, *args])

        if p is None or np.random.binomial(1, p):
            return func(*_args, **kwargs)

        if len(src_attr) == 1:
            return src_attr[0]
        return src_attr

    @action
    def apply_transform_all(self, func, *args, src=None, dst=None, p=None, **kwargs):
        """ Apply a function the whole batch at once

        Parameters
        ----------
        func : callable
            a function to apply to each item from the source

        src : str or array
            the source to get data from, can be:

            - str - a component name, e.g. 'images' or 'masks'
            - array-like - a numpy-array, list, etc

        dst : str or array
            the destination to put the result in, can be:

            - None
            - str - a component name, e.g. 'images' or 'masks'
            - array-like - a numpy-array, list, etc

        p : float or None
            probability of applying transform to an element in the batch

            if not None, indices of relevant batch elements will be passed ``func``
            as a named arg ``indices``.

        args, kwargs
            other parameters passed to ``func``

        Notes
        -----
        apply_transform_all does the following::

            self.dst = func(self.src, *args, **kwargs)

        When ``p`` is passed, random indices are chosen first and then passed to ``func``::

            self.dst = func(self.src, *args, indices=random_indices, **kwargs)

        Examples
        --------

        ::

            apply_transform_all(make_masks_fn, src='images', dst='masks')
            apply_transform_all(MyBatch.make_masks, src='images', dst='masks')
            apply_transform_all(custom_crop, src='images', dst='augmented_images', p=.2)

        """
        if not isinstance(dst, str) and not isinstance(src, str):
            raise TypeError("At least one of dst and src should be attribute names, not arrays")

        if src is None:
            _args = args
        else:
            if isinstance(src, str):
                src_attr = getattr(self, src)
            else:
                src_attr = src
            _args = tuple([src_attr, *args])

        if p is not None:
            indices = np.where(np.random.binomial(1, p, len(self)))[0]
            kwargs['indices'] = indices
        tr_res = func(*_args, **kwargs)

        if dst is None:
            pass
        elif isinstance(dst, str):
            setattr(self, dst, tr_res)
        else:
            dst[:] = tr_res
        return self

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
            except KeyError:
                raise KeyError("File {} is not indexed in the received index".format(ix))

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
            print(all_errors)
            traceback.print_tb(all_errors[0].__traceback__)
            raise RuntimeError("Could not assemble the batch")
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

    @inbatch_parallel('indices', post='_assemble', target='f', dst_default='components')
    def _load_blosc(self, ix, src=None, dst=None):
        """ Load data from a blosc packed file """
        file_name = self._get_file_name(ix, src)
        with open(file_name, 'rb') as f:
            data = dill.loads(blosc.decompress(f.read()))
            components = tuple(dst or self.components)
            try:
                item = tuple(data[i] for i in components)
            except Exception as e:
                raise KeyError('Cannot find components in corresponfig file', e)
        return item

    @inbatch_parallel('indices', target='f')
    def _dump_blosc(self, ix, dst, components=None):
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
            raise ValueError('Unknown format %s' % fmt)

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
