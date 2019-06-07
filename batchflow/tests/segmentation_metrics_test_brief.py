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
labels = np.array([0, 1, 1, 0, 2, 0, 1, 1]).reshape(batch_size, image_size, image_size)
probas = np.eye(num_classes)[labels] #onehots are basically like probas, just with all 0 and a single 1

@pytest.mark.parametrize('metrics_name', metrics_names)
class TestShape:
    """
    Tests the shape of return value for all combinations of both types of metrics aggregation and predictions shapes.
    """

    @pytest.mark.parametrize('batch_agg', [None, 'mean'])
    @pytest.mark.parametrize('multi_agg', [None, 'micro', 'macro'])
    @pytest.mark.parametrize('predics', [(labels, 'labels', None), (probas, 'proba', 3)])
    def test_noaxis_agg_shape(self, metrics_name, batch_agg, multi_agg, predics):

        predictions, fmt, axis = predics

        shapes = {'None-None' : (batch_size, num_classes),
                  'mean-None' : (num_classes,),
                  'None-micro' : (batch_size,),
                  'None-macro' : (batch_size,),
                  'mean-micro' : None,
                  'mean-macro' : None}

        exp_shape = shapes['-'.join([str(batch_agg), str(multi_agg)])]

        metric = SegmentationMetricsByPixels(targets, predictions, fmt, num_classes, axis)
        res = metric.evaluate(metrics=metrics_name, agg=batch_agg, multiclass=multi_agg)
        res_shape = res.shape if isinstance(res, np.ndarray) else None

        assert res_shape == exp_shape

class TestResult:
    """
    Tests the contents of return value for all combinations of both types of metrics aggregation.
    """

    tpr = {'None-None' : np.array([1.0, 1.0, 0.0, 0.5, 1.0, 1.0]),
           'mean-None' : np.array([0.75, 1.0, 0.5]),
           'None-micro' : np.array([0.5, 0.75]),
           'None-macro' : np.array([0.66, 0.83]),
           'mean-micro' : np.array([0.62]),
           'mean-macro' : np.array([0.75])}

    fpr = {'None-None' : np.array([0.33, 0.33, 0.0, 0.0, 0.0, 0.25]),
           'mean-None' : np.array([0.16, 0.16, 0.12]),
           'None-micro' : np.array([0.25, 0.12]),
           'None-macro' : np.array([0.22, 0.08]),
           'mean-micro' : np.array([0.18]),
           'mean-macro' : np.array([0.15])}

    tnr = {'None-None' : np.array([0.66, 0.66, 1.0, 1.0, 1.0, 0.75]),
           'mean-None' : np.array([0.83, 0.83, 0.87]),
           'None-micro' : np.array([0.75, 0.87]),
           'None-macro' : np.array([0.77, 0.91]),
           'mean-micro' : np.array([0.81]),
           'mean-macro' : np.array([0.84])}

    fnr = {'None-None' : np.array([0.0, 0.0, 1.0, 0.5, 0.0, 0.0]),
          'mean-None' : np.array([0.25, 0.0, 0.5]),
          'None-micro' : np.array([0.5, 0.25]),
          'None-macro' : np.array([0.33, 0.16]),
          'mean-micro' : np.array([0.37]),
          'mean-macro' : np.array([0.25])}

    @pytest.mark.parametrize('batch_agg', [None, 'mean'])
    @pytest.mark.parametrize('multi_agg', [None, 'micro', 'macro'])
    @pytest.mark.parametrize('metrics_name, metrics_value',
                             [('true_positive_rate', tpr), ('false_positive_rate', fpr),
                              ('true_negative_rate', tnr), ('false_negative_rate', fnr)])
    def test_contents(self, batch_agg, multi_agg, metrics_name, metrics_value):

        exp = metrics_value['-'.join([str(batch_agg), str(multi_agg)])]
        metric = SegmentationMetricsByPixels(targets, labels, 'labels', num_classes)
        res = metric.evaluate(metrics=metrics_name, agg=batch_agg, multiclass=multi_agg)
        res = res.reshape(-1) if isinstance(res, np.ndarray) else [res]

        assert np.allclose(res, exp, 1e-01)