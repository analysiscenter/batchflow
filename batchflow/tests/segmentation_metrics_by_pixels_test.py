"""
Tests for SegmentationMetricsByPixels class from batchflow.models.metrics.
Structurally, file consists of two main classes. First one tests metric's output shape, second — its contents.
Main test functions run for almost all metrics values, declared in metrics_dict (exceptions described below).
Test data is pre-defined, it's shape and contents were chosen for reasons of balance between visual simplicity
and test coverage diversity.
"""
# pylint: disable=import-error, no-name-in-module
import numpy as np
import pytest

from batchflow.models.metrics import SegmentationMetricsByPixels

# Accuracy is not included because it can't process 'multiclass' parameter and therefore is being tested individually.
METRICS_LIST = ['tpr', 'fpr', 'fnr', 'tnr', 'prv', 'ppv', 'fdr', 'for', 'npv', 'plr', 'nlr', 'dor', 'f1s', 'jac']

BATCH_SIZE = 2
IMAGE_SIZE = 2
NUM_CLASSES = 3

# Set targets.
TARGETS = np.array([0, 1, 2, 2, 0, 0, 1, 1]).reshape(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE)
# Set predictions as 'labels'.
LABELS = np.array([0, 1, 1, 0, 2, 0, 1, 1]).reshape(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE)
# Onehots are basically like probas, just with all 0 and a single 1.
PROBA = np.eye(NUM_CLASSES)[LABELS]
# Logit function gives ±infs on degenerate case of 0s and 1s, but it's okay for sigmoid function.
LOGITS = np.log(PROBA / (1. - PROBA))
# First param stands for predictions variable, second — for predictions type, third — for axis with class info.
# Transposed predictions correspond to 'channels_first' data format.
PREDICTIONS = [(LABELS, 'labels', None),
               (PROBA, 'proba', 3),
               (LOGITS, 'logits', 3),
               (np.transpose(PROBA, (3, 0, 1, 2)), 'proba', 0),
               (np.transpose(LOGITS, (3, 0, 1, 2)), 'logits', 0)]

class TestShape:
    """
    This class checks the shape of metrics' return value for all combinations of both aggregation types.

    There is a following pattern in both tests:
    0. Each function is preceded by data for it's parametrization.
    1. Parametrizing decorators are applied.
    2. Instance of SegmentationMetricsByPixels is being created.
    3. Metric is being evaluated with given parameters.
    4. It's result's shape is being compared with expected one.
    """

    # First param stands for batch aggregation, second — for multiclass one, third represents expected output shape.
    params = [(None, None, (BATCH_SIZE, NUM_CLASSES)),
              (None, 'micro', (BATCH_SIZE,)),
              (None, 'macro', (BATCH_SIZE,)),
              ('mean', None, (NUM_CLASSES,)),
              ('mean', 'micro', None),
              ('mean', 'macro', None)]

    @pytest.mark.parametrize('metric_name', METRICS_LIST)
    @pytest.mark.parametrize('predictions, fmt, axis', PREDICTIONS)
    @pytest.mark.parametrize('batch_agg, multi_agg, exp_shape', params)
    def test_shape(self, metric_name, predictions, fmt, axis, batch_agg, multi_agg, exp_shape):
        """
        Function compares expected return value shape with actual return value shape
        of metric evaluation with given params for all metrics from METRICS_LIST.

        Parameters
        ----------
        predictions : np.array
            Variable name containing predictions' array of desired format

        fmt : string
            Denotes predictions format

        axis : None or int
            A class axis

        batch_agg : string
            Cross-batch aggregation type

        multi_agg : string
            Multiclass agregation type

        exp_shape : None or tuple
            Expected return value shape
        """
        metric = SegmentationMetricsByPixels(TARGETS, predictions, fmt, NUM_CLASSES, axis)
        res = metric.evaluate(metrics=metric_name, agg=batch_agg, multiclass=multi_agg)
        res_shape = res.shape if isinstance(res, np.ndarray) else None

        assert res_shape == exp_shape

    # Individual test params for accuracy — batch-only aggregation param and corresponding expected shapes.
    params_accuracy = [(None, (BATCH_SIZE,)),
                       ('mean', None)]

    @pytest.mark.parametrize('predictions, fmt, axis', PREDICTIONS)
    @pytest.mark.parametrize('batch_agg, exp_shape', params_accuracy)
    def test_shape_accuracy(self, predictions, fmt, axis, batch_agg, exp_shape):
        """
        Function compares expected return value shape with actual return value shape
        of accuracy metric evaluation with given params.

        Parameters
        ----------
        predictions : np.array
            Variable name containing predictions' array of desired format

        fmt : string
            Denotes predictions format

        axis : None or int
            A class axis

        batch_agg : string
            Cross-batch aggregation type

        exp_shape : None or tuple
            Expected return value shape
        """
        metric = SegmentationMetricsByPixels(TARGETS, predictions, fmt, NUM_CLASSES, axis)
        res = metric.evaluate(metrics='accuracy', agg=batch_agg)
        res_shape = res.shape if isinstance(res, np.ndarray) else None

        assert res_shape == exp_shape

class TestResult:
    """
    This class checks the contents of metrics' return value for all combinations of both aggregation types.

    There is a following pattern in both tests:
    0. Each function is preceded by data for it's parametrization.
    1. Parametrizing decorators are applied.
    2. Instance of SegmentationMetricsByPixels is being created.
    3. Metric is being evaluated with given parameters.
    4. It's result is being compared with expected one.
    """

    # First param stands for batch aggregation type, second — for multiclass one,
    # third represents manually pre-calculated expected output contents for each type of metrics.
    params = [(None, None, {'tpr' : np.array([1.00, 1.00, 0.00, 0.50, 1.00, 1.00]),
                            'fpr' : np.array([0.33, 0.33, 0.00, 0.00, 0.00, 0.25]),
                            'tnr' : np.array([0.66, 0.66, 1.00, 1.00, 1.00, 0.75]),
                            'fnr' : np.array([0.00, 0.00, 1.00, 0.50, 0.00, 0.00]),
                            'prv' : np.array([0.25, 0.25, 0.50, 0.50, 0.50, 0.00]),
                            'ppv' : np.array([0.50, 0.50, 1.00, 1.00, 1.00, 0.00]),
                            'fdr' : np.array([0.50, 0.50, 0.00, 0.00, 0.00, 1.00]),
                            'for' : np.array([0.00, 0.00, 0.50, 0.33, 0.00, 0.00]),
                            'npv' : np.array([1.00, 1.00, 0.50, 0.66, 1.00, 1.00]),
                            'plr' : np.array([3.00, 3.00, 0.00, np.inf, np.inf, 4.00]),
                            'nlr' : np.array([0.00, 0.00, 1.00, 0.50, 0.00, 0.00]),
                            'dor' : np.array([np.inf, np.inf, 0.00, np.inf, np.inf, np.inf]),
                            'f1s' : np.array([0.66, 0.66, 0.00, 0.66, 1.00, 0.00]),
                            'jac' : np.array([0.49, 0.49, 0.00, 0.49, 1.00, 0.00])}),

              (None, 'micro', {'tpr' : np.array([0.50, 0.75]),
                               'fpr' : np.array([0.25, 0.12]),
                               'tnr' : np.array([0.75, 0.87]),
                               'fnr' : np.array([0.50, 0.25]),
                               'prv' : np.array([0.33, 0.33]),
                               'ppv' : np.array([0.50, 0.75]),
                               'fdr' : np.array([0.50, 0.25]),
                               'for' : np.array([0.25, 0.12]),
                               'npv' : np.array([0.75, 0.87]),
                               'plr' : np.array([3.00, 10.00]),
                               'nlr' : np.array([0.42, 0.18]),
                               'dor' : np.array([6.00, np.inf]),
                               'f1s' : np.array([0.50, 0.75]),
                               'jac' : np.array([0.33, 0.60])}),

              (None, 'macro', {'tpr' : np.array([0.66, 0.83]),
                               'fpr' : np.array([0.22, 0.08]),
                               'tnr' : np.array([0.77, 0.91]),
                               'fnr' : np.array([0.33, 0.16]),
                               'prv' : np.array([0.33, 0.33]),
                               'ppv' : np.array([0.66, 0.66]),
                               'fdr' : np.array([0.33, 0.33]),
                               'for' : np.array([0.16, 0.11]),
                               'npv' : np.array([0.83, 0.88]),
                               'plr' : np.array([2.00, 4.00]),
                               'nlr' : np.array([0.33, 0.16]),
                               'dor' : np.array([0.00, np.inf]),
                               'f1s' : np.array([0.66, 0.74]),
                               'jac' : np.array([0.50, 0.58])}),

              ('mean', None, {'tpr' : np.array([0.75, 1.00, 0.50]),
                              'fpr' : np.array([0.16, 0.16, 0.12]),
                              'tnr' : np.array([0.83, 0.83, 0.87]),
                              'fnr' : np.array([0.25, 0.00, 0.50]),
                              'prv' : np.array([0.37, 0.37, 0.25]),
                              'ppv' : np.array([0.75, 0.75, 0.50]),
                              'fdr' : np.array([0.25, 0.25, 0.50]),
                              'for' : np.array([0.16, 0.00, 0.25]),
                              'npv' : np.array([0.83, 1.00, 0.75]),
                              'plr' : np.array([3.00, 3.00, 2.00]),
                              'nlr' : np.array([0.25, 0.00, 0.50]),
                              'dor' : np.array([np.inf, np.inf, 0.00]),
                              'f1s' : np.array([0.66, 0.83, 0.00]),
                              'jac' : np.array([0.50, 0.75, 0.00])}),

              ('mean', 'micro', {'tpr' : np.array([0.62]),
                                 'fpr' : np.array([0.18]),
                                 'tnr' : np.array([0.81]),
                                 'fnr' : np.array([0.37]),
                                 'prv' : np.array([0.33]),
                                 'ppv' : np.array([0.62]),
                                 'fdr' : np.array([0.37]),
                                 'for' : np.array([0.18]),
                                 'npv' : np.array([0.81]),
                                 'plr' : np.array([6.50]),
                                 'nlr' : np.array([0.30]),
                                 'dor' : np.array([6.00]),
                                 'f1s' : np.array([0.62]),
                                 'jac' : np.array([0.46])}),

              ('mean', 'macro', {'tpr' : np.array([0.75]),
                                 'fpr' : np.array([0.15]),
                                 'tnr' : np.array([0.84]),
                                 'fnr' : np.array([0.25]),
                                 'prv' : np.array([0.33]),
                                 'ppv' : np.array([0.66]),
                                 'fdr' : np.array([0.33]),
                                 'for' : np.array([0.13]),
                                 'npv' : np.array([0.86]),
                                 'plr' : np.array([3.00]),
                                 'nlr' : np.array([0.25]),
                                 'dor' : np.array([0.00]),
                                 'f1s' : np.array([0.70]),
                                 'jac' : np.array([0.54])})]

    @pytest.mark.parametrize('predictions, fmt, axis', PREDICTIONS)
    @pytest.mark.parametrize('batch_agg, multi_agg, exp_dict', params)
    def test_contents(self, predictions, fmt, axis, batch_agg, multi_agg, exp_dict):
        """
        Function compares expected return value contents with contents of actual return value
        of metric evaluation with given params for all metrics from METRICS_DICT.

        Parameters
        ----------
        predictions : np.array
            Variable name containing predictions' array of desired format

        fmt : string
            Denotes predictions format

        axis : None or int
            A class axis

        batch_agg : string
            Cross-batch aggregation type

        multi_agg : string
            Multiclass agregation type

        exp_dict : dict
            Keys are metric's aliases and values are expected contents
            of their evaluation results with given aggregation params
        """
        metric = SegmentationMetricsByPixels(TARGETS, predictions, fmt, NUM_CLASSES, axis)
        for metric_name, exp in exp_dict.items():
            res = metric.evaluate(metrics=metric_name, agg=batch_agg, multiclass=multi_agg)
            res = res.reshape(-1) if isinstance(res, np.ndarray) else [res]

            assert np.allclose(res, exp, atol=1e-02, rtol=0), 'failed on metric {}'.format(metric_name)

    # Individual test params for accuracy — batch-only aggregation param and corresponding expected metrics contents.
    params_accuracy = [(None, np.array([0.50, 0.75])),
                       ('mean', np.array([0.62]))]

    @pytest.mark.parametrize('predictions, fmt, axis', PREDICTIONS)
    @pytest.mark.parametrize('batch_agg, exp', params_accuracy)
    def test_contents_accuracy(self, predictions, fmt, axis, batch_agg, exp):
        """
        Function compares expected return value contents with contents of actual return value
        of accuracy metric evaluation with given params.

        Parameters
        ----------
        predictions : np.array
            Variable name containing predictions' array of desired format

        fmt : string
            Denotes predictions format

        axis : None or int
            A class axis

        batch_agg : string
            Cross-batch aggregation type

        exp : np.array
            Expected contents of accuracy evaluation results with given aggregation params
        """
        metric = SegmentationMetricsByPixels(TARGETS, predictions, fmt, NUM_CLASSES, axis)
        res = metric.evaluate(metrics='accuracy', agg=batch_agg)
        res = res.reshape(-1) if isinstance(res, np.ndarray) else [res]

        assert np.allclose(res, exp, atol=1e-02, rtol=0), 'failed on metric {}'.format('accuracy')
