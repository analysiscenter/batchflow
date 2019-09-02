""" Tests for regression metrics """
import sys
import pytest

sys.path.append('../..')

import numpy as np

from batchflow.models.metrics import RegressionMetrics

TARGETS = [[1, 1], [2, 2]]
PREDICTIONS = [[1, 0], [4, 2]]
WEIGHTS = [1, 3]

RESULTS = [(None, {'mae': [1.5, 0.25],
                   'mse': [3, 0.25],
                   'median_absolute_error': [1, 0.5],
                   'max_error': [2, 1],
                   'rmse': [1.73, 0.5],
                   'r2': [-15, -0.33],
                   'explained_variance_ratio': [-3, 0],
                   'acc': [1, 1]}),

           ('mean', {'mae': 0.875,
                     'mse': 1.625,
                     'median_absolute_error': 0.75,
                     'max_error': 1.5,
                     'rmse': 1.12,
                     'r2': -7.67,
                     'explained_variance_ratio': -1.5,
                     'acc': 1})]


@pytest.mark.parametrize('output_agg, exp_res', RESULTS)
def test_values(output_agg, exp_res):
    """ Test that expected values match the real ones """
    metrics = RegressionMetrics(TARGETS, PREDICTIONS, weights=WEIGHTS, multi=True)
    for metric_name, exp in exp_res.items():
        res = metrics.evaluate(metrics=metric_name, agg=output_agg)
        assert np.allclose(res, exp, atol=0.01, rtol=0), 'failed on metric {}'.format(metric_name)
