import numpy as np
import pytest
from batchflow.models.metrics import SegmentationMetricsByPixels, SegmentationMetricsByInstances

metrics_names = ['true_positive_rate', 'sensitivity', 'recall', 'false_positive_rate', 'fallout',
                 'false_negative_rate', 'miss_rate', 'true_negative_rate', 'specificity', 
                 'positive_predictive_value', 'precision', 'false_discovery_rate', 'false_omission_rate',
                 'negative_predictive_value', 'f1_score', 'dice', 'jaccard', 'iou']
                 # accuracy() can't process 'multiclass' parameter and therefore should be tested individualy
                 # prevalence(), positive_likelihood_ratio(), negative_likelihood_ratio(), diagnostics_odds_ratio() 
                 # are missing 'when_zero' parameter and therefore should be tested individualy

sample_length = 100
num_classes = 10
targets_sample = np.random.randint(0, num_classes - 2, sample_length)
noise_sample = np.random.binomial(1, 1/10, sample_length)
predics_sample = targets_sample + noise_sample

batch_size = 3
image_size = 2
pull_out = lambda s,x,y: s[:x*y*y].reshape(x, y, y)
targets = pull_out(targets_sample, batch_size, image_size)
predics = pull_out(predics_sample, batch_size, image_size)

@pytest.mark.parametrize('metrics_name', metrics_names)
class TestShape:
    """
    Return value shape tests for both types of metrics aggregation.
    """

    aggregation_params = [(None, None, (batch_size, num_classes)),
                          ('mean', None, (num_classes,)),
                          (None, 'macro', (batch_size,)),
                          ('mean', 'macro', None)]

    @pytest.mark.parametrize('batch_agg, multi_agg, exp_shape', aggregation_params)
    def test_noaxis_agg_shape(self, metrics_name, batch_agg, multi_agg, exp_shape):
        metric = SegmentationMetricsByPixels(targets, predics, 'labels', num_classes)
        res = metric.evaluate(metrics=metrics_name, agg=batch_agg, multiclass=multi_agg)
        res_shape = res.shape if isinstance(res, np.ndarray) else None
        assert res_shape == exp_shape

    @pytest.mark.parametrize('batch_agg, multi_agg, exp_shape', aggregation_params)
    def test_axis_agg_shape(self, metrics_name, batch_agg, multi_agg, exp_shape):
        onehots = np.eye(num_classes)[predics] # convert labels to onehots (a la proba)
        axis = 3
        metric = SegmentationMetricsByPixels(targets, onehots, 'proba', num_classes, axis)
        res = metric.evaluate(metrics=metrics_name, agg=batch_agg, multiclass=multi_agg)
        res_shape = res.shape if isinstance(res, np.ndarray) else None
        assert res_shape == exp_shape

# @pytest.mark.parametrize('metrics_name', ['recall'])
class TestResult:
    """
    Return value tests for both types of metrics aggregation.
    """

    aggregation_params = [(None, None, np.array([1. , 1. , 0. , 0.5, 1. , 1. ])),
                          ('mean', None, np.array([0.75, 1.  , 0.5 ])),
                          (None, 'macro', np.array([0.66666667, 0.83333333])),
                          ('mean', 'macro', [0.75])]

    @pytest.mark.parametrize('batch_agg, multi_agg, exp', aggregation_params)
    def test_recall(self, batch_agg, multi_agg, exp):

        targets = np.array([0, 1, 2, 2, 0, 0, 1, 1]).reshape(2,2,2)
        predics = np.array([0, 1, 1, 0, 2, 0, 1, 1]).reshape(2,2,2)
        metric = SegmentationMetricsByPixels(targets, predics, 'labels', 3)

        res = metric.evaluate('recall', agg=batch_agg, multiclass=multi_agg)
        res = res.reshape(-1) if isinstance(res, np.ndarray) else [res]

        assert np.isclose(res, exp).all()
