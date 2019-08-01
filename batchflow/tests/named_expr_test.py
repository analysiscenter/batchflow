# pylint: disable=redefined-outer-name, missing-docstring, bad-continuation
import sys
from contextlib import ExitStack as does_not_raise

import pytest

sys.path.append('..')
from batchflow import B, C, D, F, V, L, R, P, I, Dataset


@pytest.mark.parametrize('named_expr', [
    C('option'),
    B('size'),
    D('size'),
    V('var'),
    R('normal', 0, 1),
    P('normal', 0, 1),
    F(lambda batch: 0),
    L(lambda: 0),
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

NAMES = ['c', 'm', 'r'] * 4
EXPECTATIONS = ([does_not_raise()] * 5 + [pytest.raises(ValueError)]) * 2
LIMIT_NAME = ['n_epochs'] * 6 + ['n_iters'] * 6
LIMIT_VALUE = [1] * 3 + [None] * 3 + [5] * 3 + [None] * 3
RESULTS = [1, 5, .2, 1, None, -1, 1, 5, .2, 1, None, -1]

@pytest.mark.parametrize('name,expectation,limit_name,limit_value,result',
                         list(zip(NAMES, EXPECTATIONS, LIMIT_NAME, LIMIT_VALUE, RESULTS))
)
def test_i(name, expectation, limit_name, limit_value, result):
    """Test checks for behaviour of I under different pipeline configurations.

    name
        Name of I, defines its output.
    expectation
        Test is expected to raise an error when names requires calculaion of total iterations (e.g. for 'm')
        and this number is not defined in pipeline (limit_value is None).
    limit_name
        n_epochs or n_iters
    limit_value
        Total numer of epochs or iteration to run.
    result
        Expected output of I. If None, I is expected to raise an error.
    """
    kwargs = {'batch_size': 2, limit_name: limit_value, 'lazy': True}

    pipeline = (Dataset(10).pipeline()
        .init_variable('var', -1)
        .update_variable('var', I(name), mode='w')
        .run(**kwargs)
    )

    with expectation:
        _ = pipeline.next_batch()

    assert pipeline.get_variable('var') == result
