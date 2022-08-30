""" Tests for regression metrics.

Verifies the correctness of evaluated metrics for the cases:
    - targets and predictions are non border values.
    - targets and predictions are border values, i.e. zeros, equal values.
    - targets and predictions are single values.
"""
# pylint: disable=import-error, no-name-in-module
# pylint: disable=missing-docstring
import pytest
import numpy as np
np.seterr(divide='ignore', invalid='ignore')

from batchflow.models.metrics import RegressionMetrics

PARAMS_DEFINED = [(None, 'mae', [1.5, 0.25]),
                  (None, 'mse', [3, 0.25]),
                  (None, 'median_absolute_error', [1, 0.5]),
                  (None, 'max_error', [2, 1]),
                  (None, 'rmse', [1.73, 0.5]),
                  (None, 'r2', [-15, -0.33]),
                  (None, 'explained_variance_ratio', [-3, 0]),
                  (None, 'acc', [1, 1]),
                  ('mean', 'mae', 0.875),
                  ('mean', 'mse', 1.625),
                  ('mean', 'median_absolute_error', 0.75),
                  ('mean', 'max_error', 1.5),
                  ('mean', 'rmse', 1.12),
                  ('mean', 'r2', -7.67),
                  ('mean', 'explained_variance_ratio', -1.5),
                  ('mean', 'acc', 1)]

@pytest.mark.parametrize('targets, predictions, weights', [([[1, 1], [2, 2]], [[1, 0], [4, 2]], [1, 3])])
@pytest.mark.parametrize('output_agg, metric_name, exp_res', PARAMS_DEFINED)
def test_defined_values(targets, predictions, weights, output_agg, metric_name, exp_res):
    metrics = RegressionMetrics(targets, predictions, weights=weights, multi=True)
    res = metrics.evaluate(metrics=metric_name, agg=output_agg)
    assert np.allclose(res, exp_res, atol=0.01, rtol=0)


PARAMS = [('eq', 'mae', 0), ('eq', 'mse', 0), ('eq', 'median_absolute_error', 0), ('eq', 'max_error', 0),
          ('eq', 'rmse', 0), ('eq', 'r2', 1), ('eq', 'explained_variance_ratio', 1), ('eq', 'acc', 1),
          ('both', 'mae', 0), ('both', 'r2', np.nan), ('both', 'median_absolute_error', 0), ('both', 'max_error', 0),
          ('both', 'rmse', 0), ('both', 'mse', 0), ('both', 'explained_variance_ratio', np.nan), ('both', 'acc', 1),
          ('single', 'r2', -np.inf), ('single', 'explained_variance_ratio', -np.inf)]

@pytest.mark.parametrize('multi', [False, True])
@pytest.mark.parametrize('mode, metric_name, exp_res', PARAMS)
def test_both_zero(multi, mode, metric_name, exp_res):
    size = (10, 2) if multi else 10
    exp_res = [exp_res, exp_res] if multi else exp_res

    if mode == 'eq':
        targets = np.arange(np.prod(size)).reshape(size)
        predictions = targets
    elif mode == 'both':
        targets = np.zeros(shape=size)
        predictions = targets
    else:
        targets = np.zeros(shape=size)
        predictions = np.arange(np.prod(size)).reshape(size)

    metrics = RegressionMetrics(targets, predictions, multi=multi)
    res = metrics.evaluate(metrics=metric_name, agg=None)
    res = list(res) if multi else res

    if np.isnan(exp_res).all():
        assert np.isnan(res).all()
    else:
        assert res == exp_res, f'failed on targets: {targets}, predictions: {predictions}'

PARAMS_SINGLE = [('mae', 1), ('mse', 1), ('median_absolute_error', 1), ('max_error', 1),
                 ('rmse', 1), ('r2', -np.inf), ('explained_variance_ratio', np.nan), ('acc', 1)]

@pytest.mark.parametrize('agg', [None, 'mean'])
@pytest.mark.parametrize('targets, predictions', [([1, 1], [2, 2])])
@pytest.mark.parametrize('metric_name, exp_res', PARAMS_SINGLE)
def test_single_value(agg, targets, predictions, metric_name, exp_res):
    exp_res = [exp_res, exp_res] if agg is None else exp_res
    metrics = RegressionMetrics(targets, predictions, multi=True)
    res = metrics.evaluate(metrics=metric_name, agg=agg)
    res = list(res) if agg is None else res
    if np.isnan(exp_res).all():
        assert np.isnan(res).all()
    else:
        assert res == exp_res
