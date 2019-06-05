import numpy as np
import pytest
from batchflow.models.metrics import SegmentationMetricsByPixels, SegmentationMetricsByInstances

metrics_names = ['true_positive_rate', 'sensitivity', 'recall', 'false_positive_rate', 'fallout',
                 'false_negative_rate', 'miss_rate', 'true_negative_rate', 'specificity', 'positive_predictive_value',
                 'precision', 'false_discovery_rate', 'false_omission_rate', 'negative_predictive_value', 'f1_score',
                 'dice', 'jaccard', 'iou', 'prevalence', 'positive_likelihood_ratio', 'negative_likelihood_ratio',
                 'diagnostics_odds_ratio']
                 # accuracy() can't process 'multiclass' parameter and therefore should be tested individualy
                 # prevalence(), positive_likelihood_ratio(), negative_likelihood_ratio(), diagnostics_odds_ratio() 
                 # were missing 'when_zero' parameter and therefore it was added in their prototypes

batch_size = 2
image_size = 2
num_classes = 3
targets = np.array([0, 1, 2, 2, 0, 0, 1, 1]).reshape(batch_size, image_size, image_size)
predics = np.array([0, 1, 1, 0, 2, 0, 1, 1]).reshape(batch_size, image_size, image_size)
onehots = np.eye(num_classes)[predics] #it's basically like probs, just with all 0 and a single 1

@pytest.mark.parametrize('metrics_name', metrics_names)
class TestShape:
    """
    Tests the shape of return value for both types of metrics aggregation.
    """

    aggregation_params = [(None, None, (batch_size, num_classes)),
                          ('mean', None, (num_classes,)),
                          (None, 'micro', (batch_size,)),
                          (None, 'macro', (batch_size,)),
                          ('mean', 'micro', None),
                          ('mean', 'macro', None)]

    @pytest.mark.parametrize('batch_agg, multi_agg, exp_shape', aggregation_params)
    def test_noaxis_agg_shape(self, metrics_name, batch_agg, multi_agg, exp_shape):

        metric = SegmentationMetricsByPixels(targets, predics, 'labels', num_classes)
        res = metric.evaluate(metrics=metrics_name, agg=batch_agg, multiclass=multi_agg)
        res_shape = res.shape if isinstance(res, np.ndarray) else None

        assert res_shape == exp_shape

    @pytest.mark.parametrize('batch_agg, multi_agg, exp_shape', aggregation_params)
    def test_axis_agg_shape(self, metrics_name, batch_agg, multi_agg, exp_shape):

        metric = SegmentationMetricsByPixels(targets, onehots, 'proba', num_classes, axis=3)
        res = metric.evaluate(metrics=metrics_name, agg=batch_agg, multiclass=multi_agg)
        res_shape = res.shape if isinstance(res, np.ndarray) else None

        assert res_shape == exp_shape

class TestResult:
    """
    Tests the contents of return value for both types of metrics aggregation.
    """

    # tpr == true positive rate
    tpr_aggregation_params = [(None, None, np.array([1.0, 1.0, 0.0, 0.5, 1.0, 1.0])),
                                 ('mean', None, np.array([0.75, 1.0, 0.5])),
                                 (None, 'micro', np.array([0.5, 0.75])),
                                 (None, 'macro', np.array([0.66, 0.83])),
                                 ('mean', 'micro', [0.62]),
                                 ('mean', 'macro', [0.75])]

    @pytest.mark.parametrize('batch_agg, multi_agg, exp', tpr_aggregation_params)
    def test_tpr(self, batch_agg, multi_agg, exp):

        metric = SegmentationMetricsByPixels(targets, predics, 'labels', num_classes)
        res = metric.evaluate('true_positive_rate', agg=batch_agg, multiclass=multi_agg)
        res = res.reshape(-1) if isinstance(res, np.ndarray) else [res]

        assert np.allclose(res, exp, 1e-01)

    # fpr == false positive rate
    fpr_aggregation_params = [(None, None, np.array([0.33, 0.33, 0.0, 0.0, 0.0, 0.25])),
                                 ('mean', None, np.array([0.16, 0.16, 0.12])),
                                 (None, 'micro', np.array([0.25, 0.12])),
                                 (None, 'macro', np.array([0.22, 0.08])),
                                 ('mean', 'micro', [0.18]),
                                 ('mean', 'macro', [0.15])]

    @pytest.mark.parametrize('batch_agg, multi_agg, exp', fpr_aggregation_params)
    def test_fpr(self, batch_agg, multi_agg, exp):

        metric = SegmentationMetricsByPixels(targets, predics, 'labels', num_classes)
        res = metric.evaluate('false_positive_rate', agg=batch_agg, multiclass=multi_agg)
        res = res.reshape(-1) if isinstance(res, np.ndarray) else [res]

        assert np.allclose(res, exp, 1e-01)