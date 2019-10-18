""" Contains classes to handle batch data components """
import copy as cp
import numpy as np
try:
    import pandas as pd
except ImportError:
    import _fake as pd


class _ADict(dict):
    """ dict that supports advanced indexing """
    def __getitem__(self, item):
        if isinstance(item, (list, np.ndarray)):
            d = dict()
            for i in item:
                d[i] = self[i]
            return d
        return super().__getitem__(item)


class BaseComponents:
    """ Base class for a components storage """
    def __init__(self, components=None, data=None, indices=None, crop=False, copy=False, cast_to_array=True):
        self.components = components
        self.data = data.data if isinstance(data, BaseComponents) else data
        self.indices = indices
        self.cast_to_array = cast_to_array
        if crop:
            self.crop(copy=copy)

    def __str__(self):
        s = str(type(self)) + ':\n'
        for comp in self.components:
            d = getattr(self, comp, None)
            s += '  ' + comp + ': ' + str(d) + '\n'
        if self.indices is not None:
            s += 'indices: ' + str(self.indices) + '\n'
        s += '  data: ' + str(self.data) +'\n'
        return s

    def as_list(self, components=None):
        """ Return components data as a tuple """
        components = tuple(components or self.components)
        return [getattr(self, comp) for comp in components]

    def as_tuple(self, components=None):
        """ Return components data as a tuple """
        return tuple(self.as_list(components))

    def as_dict(self, components=None):
        """ Return components data as a dict """
        components = tuple(components or self.components)
        return dict(zip(components, self.as_tuple(components)))

    def __len__(self):
        return len(self.data)

    def __add__(self, other):
        if not isinstance(other, tuple):
            raise TypeError("Tuple is expected, while got %s" % type(other))
        self.data = self.data + other

    def __getitem__(self, item):
        if self.components is None:
            return self.data[item]
        return type(self)(data=self.data, indices=item, crop=False)

    def __setitem__(self, item, value):
        if self.components is None:
            self.data[item] = value
        else:
            raise NotImplementedError('Item assignment is not implemented.')

    def _get_from(self, data, copy):
        if isinstance(data, dict):
            data = _ADict(data)
        if data is not None and self.indices is not None:
            data = data[self.indices]
        if copy:
            data = cp.deepcopy(data)
        return data


class ComponentsTuple(BaseComponents):
    """ Components storage for tuple-like data """
    def crop(self, copy=False):
        """ Crops from data in accordance with indices """
        if self.data is not None and self.indices is not None:
            new_data = []
            for data in self.data:
                data = self._get_from(data, copy)
                new_data.append(data)
            self.data = new_data
            self.indices = None

    def __getattr__(self, name):
        if self.components is not None and name not in self.components:
            raise AttributeError("%s does not have an attribute '%s'" % (type(self), name))

        ix = self.components.index(name)
        res = self.data[ix] if ix < len(self.data) else None
        if res is not None and self.indices is not None:
            res = res[self.indices]
        return res

    def __setattr__(self, name, value):
        if name in ('components', 'data'):
            super().__setattr__(name, value)
        elif self.components is not None and name in self.components:
            ix = self.components.index(name)

            if self.indices is None:
                new_data = list(self.data) if self.data is not None else []
                new_data = new_data + [None for _ in range(max(len(self.components) - len(new_data), 0))]
                new_data[ix] = value
                self.data = new_data
            else:
                self.data[ix][self.indices] = value
        else:
            super().__setattr__(name, value)


class ComponentsDict(BaseComponents):
    """ Components storage for dict-like data """
    def crop(self, copy=False):
        """ Crops from data in accordance with indices """
        if self.data is not None and self.indices is not None:
            new_data = {}
            components = self.components or list(self.data.keys())
            for comp in components:
                data = self.data.get(comp, None)
                data = self._get_from(data, copy)
                new_data[comp] = data
            self.data = new_data
            self.indices = None

    def __getattr__(self, name):
        if self.components is not None and name not in self.components:
            raise AttributeError("%s does not have an attribute '%s'" % (type(self), name))

        res = self.data[name]
        if res is not None and self.indices is not None:
            res = res[self.indices]
        if self.cast_to_array and isinstance(res, pd.Series):
            res = res.values
        return res

    def __setattr__(self, name, value):
        if name in ('components', 'data'):
            super().__setattr__(name, value)
        elif self.components is not None and name in self.components:
            if self.data is None:
                self.data = {}
            if self.indices is None:
                self.data[name] = value
            else:
                self.data[name][self.indices] = value
        else:
            super().__setattr__(name, value)


def use_source(components, data=None, indices=None, crop=False, copy=False, cast_to_array=True):
    """ Return data source (and make a crop and a copy if necessary) """
    _ = components, crop, cast_to_array
    if indices is not None:
        data = data[indices] if data is not None else None
    if copy and data is not None:
        data = cp.deepcopy(data)
    return data


def create_item_class(components, data=None, indices=None, crop=False, copy=False, cast_to_array=True):
    """ Create components class """
    if data is None and components is not None:
        # default components storage
        item_class = ComponentsDict
    elif isinstance(data, (dict, pd.DataFrame, ComponentsDict)):
        item_class = ComponentsDict
    elif isinstance(data, (list, tuple, ComponentsTuple)):
        item_class = ComponentsTuple
    else:
        # source is a memory-like object (ndarray, hdf5 storage, etc)
        item_class = use_source

    item = item_class(components, data=data, indices=indices, crop=crop, copy=copy, cast_to_array=cast_to_array)

    return item
