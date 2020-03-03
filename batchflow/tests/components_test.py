# pylint: disable=missing-docstring, redefined-outer-name, import-error, no-name-in-module
import sys
import pytest
import numpy as np
import pandas as pd

sys.path.append('../..')
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
    comp0 = create_item_class(components, source)
    comp1 = create_item_class(components, comp0, i(range(12, 68), dtype, indices_type), crop=False)
    comp2 = create_item_class(components, comp1, i(range(25, 48), dtype, indices_type), crop=True)
    comp3 = create_item_class(components, comp2, i(range(32, 42), dtype, indices_type), crop=False)
    comp4 = create_item_class(components, comp3, i(range(35, 40), dtype, indices_type), crop=True)
    comp5 = create_item_class(components, comp4, i(range(37, 39), dtype, indices_type), crop=False)
    comp6 = create_item_class(components, comp5, i(38, dtype, indices_type), crop=False)
    return comp0, comp1, comp2, comp3, comp4, comp5, comp6

def common_data():
    index = np.arange(SIZE)
    images = np.arange(SIZE) + 1000
    labels = np.arange(SIZE) + 100
    components = 'images', 'labels'

    return index, components, images, labels

def tuple_of_arrays(indices_type):
    index, components, *source = common_data()
    comps = make_components(components, source, indices_type)
    return (index, ) + comps

def dict_of_arrays(indices_type):
    index, components, *source = common_data()
    source = dict(zip(components, source))
    comps = make_components(components, source, indices_type)
    return (index, ) + comps

def dataframe(indices_type, use_components=True):
    index, components, *source = common_data()
    source = pd.DataFrame(dict(zip(components, source)))
    components = components if use_components else None
    comps = make_components(components, source, indices_type)
    return (index, ) + comps

def dataframe_with_str_index(indices_type, use_components=True):
    index, components, *source = common_data()
    index = index.astype('str')
    source = pd.DataFrame(dict(zip(components, source)), index=index)
    components = components if use_components else None
    comps = make_components(components, source, 'str', indices_type)
    return (index, ) + comps

def advanced_dict(indices_type):
    index, components, *source = common_data()
    index = index.astype('str')
    source = dict({comp: dict(zip(index, source[comp_no])) for comp_no, comp in enumerate(components)})
    comps = make_components(components, source, 'str', indices_type)
    return (index, ) + comps


@pytest.mark.parametrize('indices_type', ['list', 'array'])
@pytest.mark.parametrize('source', [tuple_of_arrays, dict_of_arrays, dataframe, dataframe_with_str_index,
                                    advanced_dict])
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

    def fest_getitem_getattr(self, source, indices_type):
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


    def fest_getitem_setattr1(self, source, indices_type):
        index, full, a12_68, a25_48, a32_42, a35_40, a37_39, a38 = source(indices_type)

        a38.labels = 1000

        assert (full[index[38]].labels == 138).all()
        assert (a12_68[index[38]].labels == 138).all()
        assert (a25_48[index[38]].labels == 138).all()
        assert (a32_42[index[38]].labels == 138).all()
        assert (a35_40[index[38]].labels == 1000).all()
        assert (a37_39[index[38]].labels == 1000).all()
        assert (a38.labels == 1000).all()

    def fest_getitem_setattr2(self, source, indices_type):
        index, full, a12_68, a25_48, a32_42, a35_40, a37_39, a38 = source(indices_type)

        a32_42[index[38]].labels = 1000

        assert (full[index[38]].labels == 138).all()
        assert (a12_68[index[38]].labels == 138).all()
        assert (a25_48[index[38]].labels == 1000).all()
        assert (a32_42[index[38]].labels == 1000).all()
        assert (a35_40[index[38]].labels == 138).all()
        assert (a37_39[index[38]].labels == 138).all()
        assert (a38.labels == 138).all()

    def fest_getitem_setattr3(self, source, indices_type):
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


# ------------------
#
#   No components
#
# ------------------

def make_no_components(source, dtype='int', indices_type='array'):
    comp0 = create_item_class(None, source)
    comp1 = create_item_class(None, comp0, i(range(12, 68), dtype, indices_type))
    comp2 = create_item_class(None, comp0, i(38, dtype, indices_type))
    return comp0, comp1, comp2

def array_nc(indices_type, *_):
    index, _, *source = common_data()
    source = source[1]
    comps = make_no_components(source, indices_type)
    return (index, ) + comps

def tuple_of_arrays_nc(indices_type, *_):
    index, _, *source = common_data()
    comps = make_no_components(source, indices_type)
    return (index, ) + comps

def dict_of_arrays_nc(indices_type):
    index, components, *source = common_data()
    source = dict(zip(range(len(components)), source))
    comps = make_no_components(source, indices_type)
    return (index, ) + comps


@pytest.mark.parametrize('indices_type', ['list', 'array'])
@pytest.mark.parametrize('source', [dataframe, dataframe_with_str_index])
class TestNoComponentsName:
    def test_getattr(self, source, indices_type):
        _, full, a12_68, a25_48, a32_42, a35_40, a37_39, a38 = source(indices_type, False)

        assert (full['labels'] == np.arange(SIZE) + 100).all()
        assert (a12_68['labels'] == np.arange(12, 68) + 100).all()
        assert (a25_48['labels'] == np.arange(25, 48) + 100).all()
        assert (a32_42['labels'] == np.arange(32, 42) + 100).all()
        assert (a35_40['labels'] == np.arange(35, 40) + 100).all()
        assert (a37_39['labels'] == np.arange(37, 39) + 100).all()
        assert (a38['labels'] == 138).all()

@pytest.mark.parametrize('indices_type', ['list', 'array'])
@pytest.mark.parametrize('source', [tuple_of_arrays_nc, dict_of_arrays_nc])
class TestNoComponentsPos:
    def test_getattr(self, source, indices_type):
        _, full, a12_68, a38 = source(indices_type)

        assert (full[1] == np.arange(SIZE) + 100).all()
        assert (a12_68[1] == np.arange(12, 68) + 100).all()
        assert (a38[1] == 138).all()

@pytest.mark.parametrize('indices_type', ['list', 'array'])
@pytest.mark.parametrize('source', [array_nc])
class TestNoComponents:
    def test_getattr(self, source, indices_type):
        _, full, a12_68, a38 = source(indices_type)

        assert (full == np.arange(SIZE) + 100).all()
        assert (a12_68 == np.arange(12, 68) + 100).all()
        assert (a38 == 138).all()
