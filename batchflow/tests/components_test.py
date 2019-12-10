# pylint: disable=missing-docstring, redefined-outer-name
import pytest
import numpy as np
import pandas as pd

from batchflow.components import create_item_class
from batchflow.utils import is_iterable


SIZE = 100

def i(item, dtype='int', indices_type='list'):
    if isinstance(item, range):
        item = list(item)
    elif isinstance(item, slice):
        item = list(range(item.start, item.stop, item.step))

    if dtype == 'str':
        if is_iterable(item):
            item = [str(ix) for ix in item]
        else:
            item = str(item)

    if is_iterable(item):
        if indices_type == 'array':
            item = np.array(item)

    return item


def make_components(components, source, dtype='int', indices_type='array'):
    a0 = create_item_class(components, source)
    a1 = create_item_class(components, a0, i(range(12, 68), dtype, indices_type), crop=False)
    a2 = create_item_class(components, a1, i(range(25, 48), dtype, indices_type), crop=True)
    a3 = create_item_class(components, a2, i(range(32, 42), dtype, indices_type), crop=False)
    a4 = create_item_class(components, a3, i(range(35, 40), dtype, indices_type), crop=True)
    a5 = create_item_class(components, a4, i(range(37, 39), dtype, indices_type), crop=False)
    a6 = create_item_class(components, a5, i(38, dtype, indices_type), crop=False)
    return a0, a1, a2, a3, a4, a5, a6

def common_data(components=True):
    index = np.arange(SIZE)
    images = np.arange(SIZE) + 1000
    labels = np.arange(SIZE) + 100
    if components is True:
        components = 'images', 'labels'

    return index, components, images, labels

def tuple_of_arrays(components, indices_type):
    index, components, images, labels = common_data(components)
    source = images, labels
    comps = make_components(components, source, indices_type)
    return (index, ) + comps

def dict_of_arrays(components, indices_type):
    index, components, images, labels = common_data(components)
    source = dict(zip(components, (images, labels)))
    comps = make_components(components, source, indices_type)
    return (index, ) + comps

def dataframe(components, indices_type):
    index, components, images, labels = common_data(components)
    source = pd.DataFrame(dict(zip(components, (images, labels))))
    comps = make_components(components, source, indices_type)
    return (index, ) + comps

def dataframe_with_str_index(components, indices_type):
    index, components, images, labels = common_data(components)
    index = index.astype('str')
    source = pd.DataFrame(dict(zip(components, (images, labels))), index=index)
    comps = make_components(components, source, 'str', indices_type)
    return (index, ) + comps

def advanced_dict(components, indices_type):
    index, components, images, labels = common_data(components)
    index = index.astype('str')
    source = dict(images=dict(zip(index, images)), labels=dict(zip(index, labels)))
    comps = make_components(components, source, 'str', indices_type)
    return (index, ) + comps


@pytest.mark.parametrize('components', [None, ('images', 'labels')])
@pytest.mark.parametrize('indices_type', ['list', 'array'])
@pytest.mark.parametrize('source', [tuple_of_arrays, dict_of_arrays, dataframe, dataframe_with_str_index, advanced_dict])
class TestComponents:
    def test_getattr(self, source, indices_type):
        _, full, a12_68, a25_48, a32_42, a35_40, a37_39, a38 = source(indices_type)

        if source is advanced_dict:
            assert (np.array(list(full.labels.values())) == np.arange(SIZE) + 100).all()
        else:
            assert (full.labels == np.arange(SIZE) + 100).all()

        assert (a12_68.labels == np.arange(12, 68) + 100).all()
        assert (a25_48.labels == np.arange(25, 48) + 100).all()
        assert (a32_42.labels == np.arange(32, 42) + 100).all()
        assert (a35_40.labels == np.arange(35, 40) + 100).all()
        assert (a37_39.labels == np.arange(37, 39) + 100).all()
        assert (a38.labels == 138).all()

    def test_getitem_getattr(self, source, indices_type):
        index, full, a12_68, a25_48, a32_42, a35_40, a37_39, a38 = source(indices_type)

        assert (full[index[38]].labels == 138).all()
        assert (a12_68[index[38]].labels == 138).all()
        assert (a25_48[index[38]].labels == 138).all()
        assert (a32_42[index[38]].labels == 138).all()
        assert (a35_40[index[38]].labels == 138).all()
        assert (a37_39[index[38]].labels == 138).all()

        if index.dtype.type is np.str_:
            assert (a38[index[38]].labels == 138).all()
        else:
            with pytest.raises(TypeError):
                assert (a38[index[38]].labels == 138).all()


    def test_getitem_setattr1(self, source, indices_type):
        index, full, a12_68, a25_48, a32_42, a35_40, a37_39, a38 = source(indices_type)

        a38.labels = 1000

        assert (full[index[38]].labels == 138).all()
        assert (a12_68[index[38]].labels == 138).all()
        assert (a25_48[index[38]].labels == 138).all()
        assert (a32_42[index[38]].labels == 138).all()
        assert (a35_40[index[38]].labels == 1000).all()
        assert (a37_39[index[38]].labels == 1000).all()
        assert (a38.labels == 1000).all()

    def test_getitem_setattr2(self, source, indices_type):
        index, full, a12_68, a25_48, a32_42, a35_40, a37_39, a38 = source(indices_type)

        a32_42[index[38]].labels = 1000

        assert (full[index[38]].labels == 138).all()
        assert (a12_68[index[38]].labels == 138).all()
        assert (a25_48[index[38]].labels == 1000).all()
        assert (a32_42[index[38]].labels == 1000).all()
        assert (a35_40[index[38]].labels == 138).all()
        assert (a37_39[index[38]].labels == 138).all()
        assert (a38.labels == 138).all()

    def test_getitem_setattr3(self, source, indices_type):
        index, full, a12_68, a25_48, a32_42, a35_40, a37_39, a38 = source(indices_type)

        if source is advanced_dict:
            return

        full[index[38]].labels = 1000

        assert (full[index[38]].labels == 1000).all()
        assert (a12_68[index[38]].labels == 1000).all()
        assert (a25_48[index[38]].labels == 138).all()
        assert (a32_42[index[38]].labels == 138).all()
        assert (a35_40[index[38]].labels == 138).all()
        assert (a37_39[index[38]].labels == 138).all()
        assert (a38.labels == 138).all()

    def test_getattr_setitem1(self, source, indices_type):
        index, full, a12_68, a25_48, a32_42, a35_40, a37_39, a38 = source(indices_type)

        a32_42.labels[6] = 1000

        assert (full[index[38]].labels == 138).all()
        assert (a12_68[index[38]].labels == 138).all()
        assert (a25_48[index[38]].labels == 138).all()
        assert (a32_42[index[38]].labels == 138).all()
        assert (a35_40[index[38]].labels == 138).all()
        assert (a37_39[index[38]].labels == 138).all()
        assert (a38.labels == 138).all()
