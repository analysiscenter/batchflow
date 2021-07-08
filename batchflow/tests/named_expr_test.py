# pylint: disable=redefined-outer-name, missing-docstring
import sys
from contextlib import ExitStack as does_not_raise

import pytest
import numpy as np
import pandas as pd

sys.path.append('..')
from batchflow import (B, BA, C, D, F, V, R, P, PP, I, Dataset, Pipeline, Batch,
                       apply_parallel, inbatch_parallel, action)


#--------------------
#      COMMON
#--------------------
@pytest.mark.parametrize('named_expr', [
    C('option'),
    C('not defined', default=10),
    B('size'),
    D('size'),
    V('var'),
    R('normal', 0, 1),
    R('normal', 0, 1, size=B.size),
    F(lambda: 0),
    F(lambda x: x)(0),
])
def test_general_get(named_expr):
    pipeline = (Dataset(10).pipeline({'option': 0})
        .init_variable('var')
        .do_nothing(named_expr)
        .run(2, lazy=True)
    )

    failed = False
    try:
        _ = pipeline.next_batch()
    except KeyError:
        failed = True
    if failed:
        pytest.fail("Name does not exist")


#--------------------
#         P
#--------------------
BATCH_SIZE = 3

class MyBatch(Batch):
    components = 'images', 'masks'

    @apply_parallel
    def ap_test(self, item, param):
        if isinstance(item, tuple):
            return item[0] * param, item[1] * param
        return item * param

    @action
    @inbatch_parallel('images')
    def ip_test(self, item, param):
        return item * param


ARRAY_INIT = np.arange(BATCH_SIZE).reshape((-1, 1))

P_OPTIONS = [
    P,
    PP,
]

P_NAMED_EXPRS = [
    R('normal', 0, 1),
    R('normal', 0, 1, size=2),
    R('normal', C('mean'), C('std'), size=B.size//2),
    V('var'),
    C('option'),
]

@pytest.mark.parametrize('p_type', P_OPTIONS)
@pytest.mark.parametrize('named_expr', P_NAMED_EXPRS)
@pytest.mark.parametrize('src', [
    'images',
    ['images', 'masks'],
    ('images', 'masks'),
])
def test_apply_parallel_p(p_type, named_expr, src):
    """ Check if P() is evalauted properly """
    pipeline = (Dataset(10, MyBatch).pipeline(dict(mean=0., std=1., option=ARRAY_INIT))
        .add_namespace(np)
        .init_variable('var', ARRAY_INIT)
        .update(B.images, ARRAY_INIT)
        .update(B.masks, ARRAY_INIT)
        .ap_test(src=src, param=p_type(named_expr))
        .run(BATCH_SIZE, lazy=True)
    )

    b = pipeline.next_batch()

    if isinstance(src, str):
        assert True
    else:
        assert (b.images == b.masks).all()

@pytest.mark.parametrize('p_type', P_OPTIONS)
@pytest.mark.parametrize('named_expr', P_NAMED_EXPRS)
def test_inbatch_parallel_p(p_type, named_expr):
    """ Check if P() is evalauted properly """
    pipeline = (Dataset(10, MyBatch).pipeline(dict(mean=0., std=1., option=ARRAY_INIT))
        .add_namespace(np)
        .init_variable('var', ARRAY_INIT)
        .update(B.images, ARRAY_INIT)
        .update(B.masks, ARRAY_INIT)
        .ip_test(param=p_type(named_expr))
        .run(BATCH_SIZE, lazy=True)
    )

    _ = pipeline.next_batch()

    assert True

#--------------------
#         I
#--------------------
NAMES = ['c', 'm', 'r'] * 4
EXPECTATIONS = ([does_not_raise()] * 5 + [pytest.raises(ValueError)]) * 2
LIMIT_NAME = ['n_epochs'] * 6 + ['n_iters'] * 6
LIMIT_VALUE = [1] * 3 + [None] * 3 + [5] * 3 + [None] * 3
RESULTS = [1, 5, .2, 1, None, -1, 1, 5, .2, 1, None, -1]

@pytest.mark.parametrize('name,expectation,limit_name,limit_value,result',
                         list(zip(NAMES, EXPECTATIONS, LIMIT_NAME, LIMIT_VALUE, RESULTS))
)
def test_i(name, expectation, limit_name, limit_value, result):
    """ Check for behaviour of I under different pipeline configurations.

    name
        Name of I, defines its output.
    expectation
        Test is expected to raise an error when names requires calculaion of total iterations (e.g. for 'm')
        and this number is not defined in pipeline (limit_value is None).
    limit_name
        'n_epochs' or 'n_iters'
    limit_value
        Total numer of epochs or iteration to run.
    result
        Expected output of I. If None, I is expected to raise an error.
    """
    kwargs = {'batch_size': 2, limit_name: limit_value, 'lazy': True}

    pipeline = (Dataset(10).pipeline()
        .init_variable('var', -1)
        .update(V('var', mode='w'), I(name))
        .run(**kwargs)
    )

    with expectation:
        _ = pipeline.next_batch()

    assert pipeline.get_variable('var') == result


#--------------------
#         D
#--------------------

SIZE = [30]
N_SPLITS = [2, 3, 6, 5]

@pytest.mark.parametrize('size,n_splits',
                         list(zip(SIZE, N_SPLITS))
)
def test_d(size, n_splits):
    """Test checks for behaviour of D expression in `set_dataset` action.

    size
        size of the dataset.
    n_splits
        the number if cv folds.
    """
    dataset = Dataset(size)
    dataset.cv_split(n_splits=n_splits)

    pipeline = (Pipeline()
        .init_variable('indices', default=[])
        .update(V('indices', mode='a'), B('indices')[0])
    ) << dataset.CV(C('fold')).train

    result = list(range(size))

    for fold in range(n_splits):
        pipeline.set_config({'fold': fold})
        start = fold * (size // n_splits)
        end = (fold + 1) * (size // n_splits)

        for _ in range(2):
            pipeline.reset('vars')
            pipeline.run(1)

            assert pipeline.v('indices') == result[:start] + result[end:]


#--------------------
#         BA
#--------------------

class SomeObject:
    def __init__(self):
        self.attr = 0
        self.battr = 1
        self.item = pd.DataFrame({'item_0': [1, 2, 3],
                                  'item_1': [3, 2, 1],
                                  'item_2': [-1, -1, -1],
                                  'item_3': [-1, -1, -1]})

    def __getitem__(self, key):
        return self.item[key]

    def __setitem__(self, key, value):
        print(key, value)
        self.item[key] = np.atleast_2d(value)

@pytest.mark.parametrize('batch_length', [1, 2])
def test_ba(batch_length):
    """Test behaivour of BA named expression for all sorts of read-write options.

    batch_length
        The length of array with components.
    """
    pipeline = (Dataset(1).p
        .add_components('object', [SomeObject()]*batch_length)
        .update(BA('object').attr, BA('object').battr)
        .update(BA('object')['item_0'], [10]*batch_length)
        .update(BA('object')[['item_2', 'item_3']], BA('object')[['item_0', 'item_1']])
    )
    batch = pipeline.next_batch(1)
    assert batch.object[0].attr == 1
    assert sum(batch.object[0]['item_0'] == 10) == 3
    assert np.allclose(batch.object[0][['item_2', 'item_3']], batch.object[0][['item_0', 'item_1']])
