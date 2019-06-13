import numpy as np
import pytest
from batchflow.models.metrics import SegmentationMetricsByPixels, SegmentationMetricsByInstances

metrics = {'tpr' : ['true_positive_rate', 'sensitivity', 'recall'],
           'fpr' : ['false_positive_rate', 'fallout'],
           'fnr' : ['false_negative_rate', 'miss_rate'],
           'tnr' : ['true_negative_rate', 'specificity'],
           'prv' : ['prevalence'],
           #'acc' : ['accuracy'], # can't process 'multiclass' parameter and therefore should be tested individualy
           'ppv' : ['positive_predictive_value', 'precision'],
           'fdr' : ['false_discovery_rate'],
           'for' : ['false_omission_rate'],
           'npv' : ['negative_predictive_value'],
           'plr' : ['positive_likelihood_ratio'],
           'nlr' : ['negative_likelihood_ratio'],
           'dor' : ['diagnostics_odds_ratio'],
           'fos' : ['f1_score', 'dice'],
           'jac' : ['jaccard', 'iou']}
            # prevalence(), positive_likelihood_ratio(), negative_likelihood_ratio(), diagnostics_odds_ratio() 
            # were missing 'when_zero' parameter and therefore it was added in their prototypes

metrics_names = [name for names in list(metrics.values()) for name in names]

batch_size = 2
image_size = 2
num_classes = 3
targets = np.array([0, 1, 2, 2, 0, 0, 1, 1]).reshape(batch_size, image_size, image_size)
labels = np.array([0, 1, 1, 0, 2, 0, 1, 1]).reshape(batch_size, image_size, image_size)
proba = np.eye(num_classes)[labels] #onehots are basically like probas, just with all 0 and a single 1

@pytest.mark.parametrize('metrics_name', metrics_names)
class TestShape:
    """
    Tests the shape of return value for all combinations of both types of metrics aggregation and predictions shapes.
    """

    predictions_params = [(labels, 'labels', None),
                          (proba, 'proba', 3)]

    aggregation_params = [(None, None, (batch_size, num_classes)),
                          (None, 'micro', (batch_size,)),
                          (None, 'macro', (batch_size,)),
                          ('mean', None, (num_classes,)),
                          ('mean', 'micro', None),
                          ('mean', 'macro', None)]

    @pytest.mark.parametrize('predictions, fmt, axis', predictions_params)
    @pytest.mark.parametrize('batch_agg, multi_agg, exp_shape', aggregation_params)
    def test_shape(self, metrics_name, predictions, fmt, axis, batch_agg, multi_agg, exp_shape):

        metric = SegmentationMetricsByPixels(targets, predictions, fmt, num_classes, axis)
        res = metric.evaluate(metrics=metrics_name, agg=batch_agg, multiclass=multi_agg)
        res_shape = res.shape if isinstance(res, np.ndarray) else None

        assert res_shape == exp_shape

class TestResult:
    """
    Tests the contents of return value for all combinations of both types of metrics aggregation.
    """
    params = [(None, None, {'tpr' : np.array([1.0, 1.0, 0.0, 0.5, 1.0, 1.0]),
                            'fpr' : np.array([0.33, 0.33, 0.0, 0.0, 0.0, 0.25]),
                            'tnr' : np.array([0.66, 0.66, 1.0, 1.0, 1.0, 0.75]),
                            'fnr' : np.array([0.0, 0.0, 1.0, 0.5, 0.0, 0.0])}),

              (None, 'micro', {'tpr' : np.array([0.5, 0.75]),
                               'fpr' : np.array([0.25, 0.12]),
                               'tnr' : np.array([0.75, 0.87]),
                               'fnr' : np.array([0.5, 0.25])}),

              (None, 'macro', {'tpr' : np.array([0.66, 0.83]),
                               'fpr' : np.array([0.22, 0.08]),
                               'tnr' : np.array([0.77, 0.91]),
                               'fnr' : np.array([0.33, 0.16])}),

              ('mean', None, {'tpr' : np.array([0.75, 1.0, 0.5]),
                              'fpr' : np.array([0.16, 0.16, 0.12]),
                              'tnr' : np.array([0.83, 0.83, 0.87]),
                              'fnr' : np.array([0.25, 0.0, 0.5])}),

              ('mean', 'micro', {'tpr' : np.array([0.62]),
                                 'fpr' : np.array([0.18]),
                                 'tnr' : np.array([0.81]),
                                 'fnr' : np.array([0.37])}),

              ('mean', 'macro', {'tpr' : np.array([0.75]),
                                 'fpr' : np.array([0.15]),
                                 'tnr' : np.array([0.84]),
                                 'fnr' : np.array([0.25])})]

    @pytest.mark.parametrize('batch_agg, multi_agg, exp_dict', params)
    def test_contents(self, exp_dict, batch_agg, multi_agg):

        for alias, exp in exp_dict.items():
            metric_names = metrics[alias]
            for metric_name in metric_names:
                metric = SegmentationMetricsByPixels(targets, labels, 'labels', num_classes)
                res = metric.evaluate(metrics=metric_name, agg=batch_agg, multiclass=multi_agg)
                res = res.reshape(-1) if isinstance(res, np.ndarray) else [res]
                assert np.allclose(res, exp, 1e-01)