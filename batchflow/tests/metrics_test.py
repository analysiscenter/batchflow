"""Test SegmentationMetricsByPixels and SegmentationMetricsByInstances classes.
Also act as tests for ClassificationMetrics, since it's identical to
SegmentationMetricsByPixels.

Structurally, file consists of four classes, which respectively check:
- basic assembly process (shapes compatibility, confusion matrix corectness);
- evaluated result shape of SegmemtationMetricsByPixels for all metrics;
- similarly, evaluated result contents;
- so-called "subsampling" functions of SegmentationMetricsByInstances.

Test data is pre-defined, it's shape and contents were chosen for reasons
of balance between visual simplicity and test coverage diversity.
"""
# pylint: disable=import-error, no-name-in-module, invalid-name, protected-access
import numpy as np
import pytest

from batchflow.models.metrics import SegmentationMetricsByPixels, SegmentationMetricsByInstances

# Accuracy is not included because it can't process 'multiclass' parameter
# and therefore is being tested individually.
METRICS_LIST = ['tpr', 'fpr', 'fnr', 'tnr', 'prv', 'ppv', 'fdr', 'for', 'npv', 'plr', 'nlr', 'dor', 'f1s', 'jac']

BATCH_SIZE = 4
IMAGE_SIZE = 2
NUM_CLASSES = 3

# Set targets.
TARGETS = np.array([[[0, 1],
                     [2, 2]],

                    [[0, 0],
                     [1, 1]],

                    [[0, 1],
                     [0, 2]],

                    [[0, 0],
                     [1, 1]]
                     ])
# Set predictions as 'labels'.
LABELS = np.array([[[0, 1],
                    [1, 0]],

                   [[2, 0],
                    [1, 1]],

                    [[0, 1],
                     [2, 1]],

                    [[0, 0],
                     [0, 1]]
                     ])

# Onehots are basically like probas, just with all 0 and a single 1.
PROBA = np.eye(NUM_CLASSES)[LABELS]

# Logit function gives ±infs on degenerate case of 0s and 1s but works fine for sigmoid function.
LOGITS = np.where(PROBA > 0.5, np.inf, -np.inf)

"""First param stands for predictions variable, second — for predictions type, third — for axis with class info.
Transposed predictions correspond to 'channels_first' data format."""
PREDICTIONS = [(LABELS, 'labels', None),
               (PROBA, 'proba', 3),
               (LOGITS, 'logits', 3),
               (np.transpose(PROBA, (3, 0, 1, 2)), 'proba', 0),
               (np.transpose(LOGITS, (3, 0, 1, 2)), 'logits', 0)]

BAD_PREDICTIONS = [(LABELS[0], 'labels', None), # predictions ndim is less then targets' for labels
                   (PROBA, 'proba', None), # axis is None for multiclass proba
                   (LOGITS, 'logits', None)] # axis is None for multiclass logits

class TestAssembly:
    """Check metrics creation process."""

    @pytest.mark.parametrize('SegmentationMetrics', [SegmentationMetricsByPixels, SegmentationMetricsByInstances])
    @pytest.mark.parametrize('predictions, fmt, axis', BAD_PREDICTIONS)
    def test_incompatibility_processing(self, SegmentationMetrics, predictions, fmt, axis):
        """Create metrics class with inconsistent targets and predictions
        (different ndim, no axis when it's required), expecting ValueError.

        Parameters
        ----------
        SegmentationMetrics: SegmentationsMetricsByPixels or
                             SegmentationsMetricsByInstances
            Metrics class

        predictions : np.array
            Variable name containing predictions' array of desired format

        fmt : string
            Denotes predictions format

        axis : None or int
            A class axis
        """
        with pytest.raises(ValueError):
            SegmentationMetrics(TARGETS, predictions, fmt, NUM_CLASSES, axis)

    params = [(SegmentationMetricsByPixels, np.array([[[1, 0, 1],
                                                       [0, 1, 1],
                                                       [0, 0, 0]],

                                                      [[1, 0, 0],
                                                       [0, 2, 0],
                                                       [1, 0, 0]],

                                                      [[1, 0, 0],
                                                       [0, 1, 1],
                                                       [1, 0, 0]],

                                                      [[2, 1, 0],
                                                       [0, 1, 0],
                                                       [0, 0, 0]]]),
                                                       ),

              (SegmentationMetricsByInstances, np.array([[[[0, 0],
                                                           [1, 1]],

                                                          [[0, 1],
                                                           [0, 0]]],


                                                         [[[0, 0],
                                                           [0, 1]],

                                                          [[0, 0],
                                                           [1, 0]]],

                                                         [[[0, 0],
                                                           [0, 1]],

                                                          [[0, 1],
                                                           [1, 0]]],

                                                          [[[0, 0],
                                                           [0, 1]],

                                                          [[0, 0],
                                                           [0, 0]]],
                                                           ]))]
    @pytest.mark.parametrize('SegmentationMetrics, exp_matrix', params)
    @pytest.mark.parametrize('predictions, fmt, axis', PREDICTIONS)
    def test_confusion_matrix(self, SegmentationMetrics, exp_matrix, predictions, fmt, axis):
        """Compare contents of actual confusion matrix with expected ones
        for metrics class assembled with given params.

        Parameters
        ----------
        SegmentationMetrics: SegmentationsMetricsByPixels or
                             SegmentationsMetricsByInstances
            Metrics class

        exp_matrix: np.array
            Expected confusion matrix

        predictions : np.array
            Variable name containing predictions' array of desired format

        fmt : string
            Denotes predictions format

        axis : None or int
            A class axis
        """
        metric = SegmentationMetrics(TARGETS, predictions, fmt, NUM_CLASSES, axis)
        res_matrix = metric._confusion_matrix
        assert np.array_equal(res_matrix, exp_matrix)

class TestShape:
    """Check the shape of evaluated metrics return value for various parameters.

    There is a following pattern in both tests:
    0. Each function is preceded by data for it's parametrization.
    1. Parametrizing decorators are applied.
    2. Instance of SegmentationMetricsByPixels is being created.
    3. Metric is being evaluated with given parameters.
    4. It's result's shape is being compared with expected one.
    """

    # First param stands for batch aggregation, second — for multiclass one, third represents expected output shape.
    params = [(None, None, lambda l: (BATCH_SIZE, NUM_CLASSES - l)),
              (None, 'micro', (BATCH_SIZE,)),
              (None, 'macro', (BATCH_SIZE,)),
              ('mean', None, lambda l: (NUM_CLASSES - l,)),
              ('mean', 'micro', None),
              ('mean', 'macro', None)]
    @pytest.mark.parametrize('metric_name', METRICS_LIST)
    @pytest.mark.parametrize('predictions, fmt, axis', PREDICTIONS)
    @pytest.mark.parametrize('batch_agg, multi_agg, exp_shape', params)
    @pytest.mark.parametrize('skip_bg', [False, True])
    def test_shape(self, metric_name, predictions, fmt, axis, batch_agg, multi_agg, exp_shape, skip_bg):
        """Compare expected return value shape with actual return value shape of
        metric evaluation with given params for all metrics from METRICS_LIST.

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

        skip_bg : False or True
            If background class should be excluded from metrics evaluation
        """
        if callable(exp_shape):
            exp_shape = exp_shape(skip_bg)
        metric = SegmentationMetricsByPixels(targets=TARGETS, predictions=predictions, fmt=fmt,
                                             num_classes=NUM_CLASSES, axis=axis, skip_bg=skip_bg)
        res = metric.evaluate(metrics=metric_name, agg=batch_agg, multiclass=multi_agg)
        res_shape = res.shape if isinstance(res, np.ndarray) else None
        assert res_shape == exp_shape

    @pytest.mark.parametrize('predictions, fmt, axis', PREDICTIONS)
    @pytest.mark.parametrize('batch_agg, exp_shape', [(None, (BATCH_SIZE,)), ('mean', None)])
    def test_shape_accuracy(self, predictions, fmt, axis, batch_agg, exp_shape):
        """Compare expected return value shape with actual return value shape of
        accuracy metric evaluation with given params.

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
    """Check evaluated metrics return value for various parameters.

    There is a following pattern in both tests:
    0. Each function is preceded by data for it's parametrization.
    1. Parametrizing decorators are applied.
    2. Instance of SegmentationMetricsByPixels is being created.
    3. Metric is being evaluated with given parameters.
    4. It's result is being compared with expected one.
    """

    # First param stands for batch aggregation type, second — for multiclass one,
    # third represents manually pre-calculated expected output contents for each type of metrics.
    params = [(None, None, {'tpr' : np.array([1.00, 1.00, 0.00, 0.50, 1.00, 1.00, 0.50, 1.00, 0.00, 1.00, 0.50, 1.00]),
                            'fpr' : np.array([0.33, 0.33, 0.00, 0.00, 0.00, 0.25, 0.00, 0.33, 0.33, 0.50, 0.00, 0.00]),
                            'tnr' : np.array([0.66, 0.66, 1.00, 1.00, 1.00, 0.75, 1.00, 0.66, 0.66, 0.50, 1.00, 1.00]),
                            'fnr' : np.array([0.00, 0.00, 1.00, 0.50, 0.00, 0.00, 0.50, 0.00, 1.00, 0.00, 0.50, 0.00]),
                            'prv' : np.array([0.25, 0.25, 0.50, 0.50, 0.50, 0.00, 0.50, 0.25, 0.25, 0.50, 0.50, 0.00]),
                            'ppv' : np.array([0.50, 0.50, 1.00, 1.00, 1.00, 0.00, 1.00, 0.50, 0.00, 0.66, 1.00, 1.00]),
                            'fdr' : np.array([0.50, 0.50, 0.00, 0.00, 0.00, 1.00, 0.00, 0.50, 1.00, 0.33, 0.00, 0.00]),
                            'for' : np.array([0.00, 0.00, 0.50, 0.33, 0.00, 0.00, 0.33, 0.00, 0.33, 0.00, 0.33, 0.00]),
                            'npv' : np.array([1.00, 1.00, 0.50, 0.66, 1.00, 1.00, 0.66, 1.00, 0.66, 1.00, 0.66, 1.00]),
                            'plr' : np.array([3.00, 3.00, 0.00, np.inf, np.inf, 4.00,
                                              np.inf, 3.00, 0.00, 2.00, np.inf, np.inf]),
                            'nlr' : np.array([0.00, 0.00, 1.00, 0.50, 0.00, 0.00, 0.50, 0.00, 1.50, 0.00, 0.50, 0.00]),
                            'dor' : np.array([np.inf, np.inf, 0.00, np.inf, np.inf, np.inf,
                                              np.inf, np.inf, 0, np.inf, np.inf, np.inf]),
                            'f1s' : np.array([0.66, 0.66, 0.00, 0.66, 1.00, 0.00,
                                              0.66, 0.66, 0.00, 0.80, 0.66, np.inf]),
                            'jac' : np.array([0.50, 0.50, 0.00, 0.50, 1.00, 0.00,
                                              0.50, 0.50, 0.00, 0.66, 0.50, np.inf])}),

              (None, 'micro', {'tpr' : np.array([0.50, 0.75, 0.50, 0.75]),
                               'fpr' : np.array([0.25, 0.12, 0.25, 0.12]),
                               'tnr' : np.array([0.75, 0.87, 0.75, 0.88]),
                               'fnr' : np.array([0.50, 0.25, 0.50, 0.25]),
                               'prv' : np.array([0.33, 0.33, 0.33, 0.33]),
                               'ppv' : np.array([0.50, 0.75, 0.50, 0.75]),
                               'fdr' : np.array([0.50, 0.25, 0.50, 0.25]),
                               'for' : np.array([0.25, 0.12, 0.25, 0.12]),
                               'npv' : np.array([0.75, 0.87, 0.75, 0.88]),
                               'plr' : np.array([3.00, 10.00, 2.25, 5.00]),
                               'nlr' : np.array([0.42, 0.18, 0.64, 0.20]),
                               'dor' : np.array([6.00, np.inf, np.inf, np.inf]),
                               'f1s' : np.array([0.50, 0.75, 0.50, 0.75]),
                               'jac' : np.array([0.33, 0.60, 0.33, 0.60])}),

              (None, 'macro', {'tpr' : np.array([0.66, 0.83, 0.5, 0.83]),
                               'fpr' : np.array([0.22, 0.08, 0.22, 0.16]),
                               'tnr' : np.array([0.77, 0.91, 0.78, 0.84]),
                               'fnr' : np.array([0.33, 0.16, 0.50, 0.17]),
                               'prv' : np.array([0.33, 0.33, 0.33, 0.33]),
                               'ppv' : np.array([0.66, 0.66, 0.50, 0.88]),
                               'fdr' : np.array([0.33, 0.33, 0.50, 0.11]),
                               'for' : np.array([0.16, 0.11, 0.22, 0.11]),
                               'npv' : np.array([0.83, 0.88, 0.77, 0.88]),
                               'plr' : np.array([2.00, 4.00, 1.50, 2.00]),
                               'nlr' : np.array([0.33, 0.16, 0.66, 0.16]),
                               'dor' : np.array([0.00, np.inf, 0.00, np.inf]),
                               'f1s' : np.array([0.58, 0.71, 0.50, 0.79]),
                               'jac' : np.array([0.4, 0.55, 0.33, 0.65])}),

              ('mean', None, {'tpr' : np.array([0.75, 0.87, 0.50]),
                              'fpr' : np.array([0.21, 0.16, 0.14]),
                              'tnr' : np.array([0.79, 0.83, 0.85]),
                              'fnr' : np.array([0.25, 0.12, 0.50]),
                              'prv' : np.array([0.43, 0.37, 0.18]),
                              'ppv' : np.array([0.79, 0.75, 0.50]),
                              'fdr' : np.array([0.20, 0.25, 0.50]),
                              'for' : np.array([0.16, 0.08, 0.20]),
                              'npv' : np.array([0.83, 0.91, 0.79]),
                              'plr' : np.array([2.50, 3.00, 1.33]),
                              'nlr' : np.array([0.25, 0.12, 0.62]),
                              'dor' : np.array([np.inf, np.inf, 0.00]),
                              'f1s' : np.array([0.70, 0.74, 0.00]),
                              'jac' : np.array([0.54, 0.625, 0.00])}),

              ('mean', 'micro', {'tpr' : np.array([0.62]),
                                 'fpr' : np.array([0.18]),
                                 'tnr' : np.array([0.81]),
                                 'fnr' : np.array([0.37]),
                                 'prv' : np.array([0.33]),
                                 'ppv' : np.array([0.62]),
                                 'fdr' : np.array([0.37]),
                                 'for' : np.array([0.18]),
                                 'npv' : np.array([0.81]),
                                 'plr' : np.array([5.06]),
                                 'nlr' : np.array([0.36]),
                                 'dor' : np.array([6.00]),
                                 'f1s' : np.array([0.62]),
                                 'jac' : np.array([0.46])}),

              ('mean', 'macro', {'tpr' : np.array([0.70]),
                                 'fpr' : np.array([0.17]),
                                 'tnr' : np.array([0.82]),
                                 'fnr' : np.array([0.29]),
                                 'prv' : np.array([0.33]),
                                 'ppv' : np.array([0.68]),
                                 'fdr' : np.array([0.31]),
                                 'for' : np.array([0.15]),
                                 'npv' : np.array([0.84]),
                                 'plr' : np.array([2.37]),
                                 'nlr' : np.array([0.33]),
                                 'dor' : np.array([0.00]),
                                 'f1s' : np.array([0.64]),
                                 'jac' : np.array([0.48])})
            ]
    @pytest.mark.parametrize('predictions, fmt, axis', PREDICTIONS)
    @pytest.mark.parametrize('batch_agg, multi_agg, exp_dict', params)
    def test_result(self, predictions, fmt, axis, batch_agg, multi_agg, exp_dict):
        """Compare expected evaluated metrics return value with actual one
        with given params for all metrics from METRICS_DICT.

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

    @pytest.mark.parametrize('predictions, fmt, axis', PREDICTIONS)
    @pytest.mark.parametrize('batch_agg, exp', [(None, np.array([0.50, 0.75, 0.50, 0.75])), ('mean', np.array([0.62]))])
    def test_result_accuracy(self, predictions, fmt, axis, batch_agg, exp):
        """Compare expected evaluated metrics return value actual one
        with given params for `accuracy` metrics.

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
            Expected `accuracy` evaluation result with given aggregation params
        """
        metric = SegmentationMetricsByPixels(TARGETS, predictions, fmt, NUM_CLASSES, axis)
        res = metric.evaluate(metrics='accuracy', agg=batch_agg)
        res = res.reshape(-1) if isinstance(res, np.ndarray) else [res]
        assert np.allclose(res, exp, atol=1e-02, rtol=0), 'failed on metric {}'.format('accuracy')

class TestSubsampling:
    """Check the correctness of confusion matrix subsampling functions result
    for SegmentationMetricsByInstances class (e.g. true_positive subsample,
    total_population subsample). Test functions here act as an equivalent of
    TestResult functions for SegmentationMetricsByInstances class, since it
    differs from SegmentationMetricsByPixels in redefined subsampling functions
    (and confusion matrix assembly process, which is checked in TestAssembly).
    """

    params = [('true_positive', np.array([[1, 0],
                                          [1, 0],
                                          [1, 0],
                                          [1, 0]])),
              ('condition_positive', np.array([[1, 1],
                                               [1, 0],
                                               [1, 1],
                                               [1, 0]])),
              ('prediction_positive', np.array([[2, 0],
                                                [1, 1],
                                                [1, 1],
                                                [1, 0]])),
              ('total_population', np.array([[2, 1],
                                             [1, 1],
                                             [1, 2],
                                             [1, 0]]))]
    @pytest.mark.parametrize('subsample_name, exp_subsample', params)
    def test_subsampling(self, subsample_name, exp_subsample):
        """Compare expected subsample with actual one.

        Parameters
        ----------
        subsample_name: string
            Name of confusion matrix subsample

        exp_subsample: np.array
            Expected subsample of confusion matrix
        """
        metric = SegmentationMetricsByInstances(TARGETS, LABELS, 'labels', NUM_CLASSES)
        res_subsample = getattr(metric, subsample_name)()
        assert np.array_equal(res_subsample, exp_subsample)

    def test_subsampling_true_negative(self):
        """Check if subsampling true negative raises ValueError."""
        metric = SegmentationMetricsByInstances(TARGETS, LABELS, 'labels', NUM_CLASSES)
        with pytest.raises(ValueError):
            getattr(metric, 'true_negative')()
