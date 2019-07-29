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
    I('c'),
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

NAMES = ['c', 'm', 'r'] * 2
EXPECTATIONS = [does_not_raise()] * 4 + [pytest.raises(ValueError)] * 2
N_EPOCHS = [1] * 3 + [None] * 3
RESULTS = [1, None, None, 1, 5, .2]

@pytest.mark.parametrize('name,expectation,n_epochs,result',
                         list(zip(NAMES, EXPECTATIONS, N_EPOCHS, RESULTS))
)
def test_i(name, expectation, n_epochs, result):
    """Test checks for behaviour of I under differen pipeline configurations.

    name
        Name of I, defines its output.
    expectation
        Test is expected to raise an error when names requires calculaion of total iterations (e.g. for 'm')
        and this number is not defined in pipeline (e.g. n_epochs is None).
    n_epochs
        Total numer of epochs to run pipeline.
    result
        Expected output of I. If None, I is expected to raise an error.
    """
    pipeline = (Dataset(10).pipeline()
        .init_variable('var')
        .update_variable('var', I(name), mode='w')
        .run(2, n_epochs=n_epochs, lazy=True)
    )

    with expectation:
        _ = pipeline.next_batch()

    assert pipeline.get_variable('var') == result
