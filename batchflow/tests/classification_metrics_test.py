"""File contains various tests for classification metrics in batchflow"""
import pytest
import numpy as np
from batchflow.models.metrics import ClassificationMetrics as cm

# tests for confusion matrix

def test_confusion_matrix():
    """Test on correctness of confusion matrix calculation"""
    y_true, y_pred = np.array([1, 1, 0, 1, 0, 0]), np.array([0, 0, 1, 0, 0, 0])
    conf_matrix = np.array([[2, 3], [1, 0]])
    conf_matrix_calc = cm(y_true, y_pred, fmt='labels', num_classes=2)._confusion_matrix  #pylint:disable=protected-access
    assert (conf_matrix_calc == conf_matrix).all()

def test_confusion_matrix_multiclass():
    """Test on correctness of confusion matrix calculation for"""
    y_true, y_pred = np.array([2, 1]), np.array([[0.1, 0.1, 0.8], [0.1, 0.8, 0.1]])
    conf_matrix = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 1]])
    conf_matrix_calc = cm(y_true, y_pred, fmt='proba', axis=1, num_classes=3)._confusion_matrix  #pylint:disable=protected-access
    assert (conf_matrix_calc == conf_matrix).all()

# tests on shapes

@pytest.mark.parametrize('metrics', [
    ('accuracy'),
    ('f1_score'),
    ('true_positive_rate'),
    ('false_positive_rate')
])
def test_diff_shapes_two_classes_acc(metrics):
    """Testing different shape"""
    y_true_1, y_pred_1 = np.array([0, 1]), np.array(1)
    y_true_2, y_pred_2 = np.array([[0, 1], [1, 0]]), np.array([0, 1])
    y_true_3, y_pred_3 = np.array([[[0, 1], [1, 0]], [[1, 1], [0, 0]]]), np.array([[0, 1], [1, 0]])
    with pytest.raises(ValueError):
        getattr(cm(y_true_1, y_pred_1, fmt='labels', num_classes=2), metrics)()
    with pytest.raises(ValueError):
        getattr(cm(y_true_2, y_pred_2, fmt='labels', num_classes=2), metrics)()
    with pytest.raises(ValueError):
        getattr(cm(y_true_3, y_pred_3, fmt='labels', num_classes=2), metrics)()

@pytest.mark.parametrize('metrics', [
    ('accuracy'),
    ('f1_score'),
    ('true_positive_rate'),
    ('false_positive_rate')
])
def test_diff_shapes_multiclass_acc(metrics):
    """Testing different shape"""
    y_true_1, y_pred_1 = np.array(2), np.array([[0.8, 0.1, 0.1], [0.1, 0.1, 0.8]])
    y_true_2 = np.array([[[0, 1, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1]], [[1, 0, 0], [1, 1, 0], [1, 0, 1], [0, 0, 1]]])
    y_pred_2 = np.array([[0.1, 0.8, 0.1], [0.1, 0.8, 0.1], [0.8, 0.1, 0.1], [0.8, 0.1, 0.1]])
    with pytest.raises(ValueError):
        getattr(cm(y_true_1, y_pred_1, fmt='proba', axis=1, num_classes=3), metrics)()
    with pytest.raises(ValueError):
        getattr(cm(y_true_2, y_pred_2, fmt='proba', axis=1, num_classes=3), metrics)()

@pytest.mark.parametrize('metrics, metrics_shape', [
    ('accuracy', ()),
    ('f1_score', (1, 1)),
    ('true_positive_rate', (1, 1)),
    ('false_positive_rate', (1, 1))
])
def test_single_value_two_class(metrics, metrics_shape):
    """Test on accuracy single value in case of two class classification"""
    y_true, y_pred = np.random.choice([0, 1], size=(5,)), np.random.choice([0, 1], size=(5,))
    test_shape = getattr(cm(y_true, y_pred, fmt='labels', num_classes=2), metrics)().shape
    assert test_shape == metrics_shape

@pytest.mark.parametrize('metrics, metrics_shape', [
    ('accuracy', ()),
    ('f1_score', (1, 1)),
    ('true_positive_rate', (1, 1)),
    ('false_positive_rate', (1, 1))
])
def test_vector_multiclass(metrics, metrics_shape):
    """Test on accuracy multiclass"""
    y_true, y_pred = np.array([2, 1]), np.array([[0.1, 0.1, 0.8], [0.1, 0.8, 0.1]])
    test_shape = getattr(cm(y_true, y_pred, fmt='proba', axis=1, num_classes=2), metrics)().shape
    assert test_shape == metrics_shape

@pytest.mark.parametrize('metrics, metrics_shape', [
    ('accuracy', (3, )),
    ('f1_score', (3, 1)),
    ('true_positive_rate', (3, 1)),
    ('false_positive_rate', (3, 1))
])
def test_vector_batches_two_class(metrics, metrics_shape):
    """Test on accuracy vector with batch shape if input is a multidimensional array"""
    y_true = np.array([[[1, 1], [0, 1]], [[0, 1], [1, 1]], [[1, 0], [1, 1]]])
    y_pred = np.array([[[0, 1], [1, 1]], [[1, 0], [0, 0]], [[0, 0], [0, 1]]])
    test_shape = getattr(cm(y_true, y_pred, fmt='labels', num_classes=2), metrics)().shape
    assert test_shape == metrics_shape

@pytest.mark.parametrize('metrics, metrics_shape', [
    ('accuracy', (2, )),
    ('f1_score', (2, 1)),
    ('true_positive_rate', (2, 1)),
    ('false_positive_rate', (2, 1))
])
def test_vector_batches_multiclass(metrics, metrics_shape):
    """Test on accuracy vector with batch shape if input is a multidimensional array multiclass"""
    y_true = np.array([[[0, 1, 0], [1, 0, 0]], [[1, 0, 0], [0, 0, 1]]])
    y_pred = np.array([[[0.1, 0.8, 0.1], [0.8, 0.1, 0.1]], [[0.8, 0.1, 0.1], [0.1, 0.1, 0.8]]])
    test_shape = getattr(cm(y_true, y_pred, fmt='proba', axis=1, num_classes=3), metrics)().shape
    assert test_shape == metrics_shape

@pytest.mark.parametrize('metrics', [
    ('accuracy'),
    ('f1_score'),
    ('true_positive_rate'),
    ('false_positive_rate')
])
def test_axis_for_multiclass(metrics):
    """Test on axis=None"""
    y_true, y_pred = np.array([2, 1]), np.array([[0.1, 0.1, 0.8], [0.1, 0.8, 0.1]])
    with pytest.raises(ValueError):
        getattr(cm(y_true, y_pred, fmt='proba', axis=None, num_classes=3), metrics)()

#accuracy tests

@pytest.mark.parametrize('y_true,y_pred,acc', [
    (np.array([1, 1, 0, 1]), np.array([0, 0, 1, 0]), 0.0),
    (np.array([1, 1, 0, 1]), np.array([1, 1, 1, 0]), 0.5),
    (np.array([1, 1, 0, 1]), np.array([1, 1, 0, 1]), 1.0)
])
def test_accuracy_calculation(y_true, y_pred, acc):
    """Test on correctness of accuracy calculation"""
    assert acc == cm(y_true, y_pred, fmt='labels', num_classes=2).accuracy()

def test_accuracy_calculation_multiclass():
    """Test on correctness of accuracy calculation"""
    y_true, y_pred = np.array([2, 1]), np.array([[0.1, 0.1, 0.8], [0.1, 0.8, 0.1]])
    accuracy = 1.0
    assert accuracy == cm(y_true, y_pred, fmt='proba', axis=1, num_classes=3).accuracy()

# f1 score tests

@pytest.mark.parametrize('y_true, y_pred, f1_score', [
    (np.array([1, 1, 0, 1]), np.array([0, 0, 1, 0]), 0.0),
    (np.array([1, 1, 0, 0]), np.array([0, 1, 1, 0]), 0.5),
    (np.array([1, 1, 1, 0]), np.array([1, 1, 1, 0]), 1.0)
])
def test_f1_calculation(y_true, y_pred, f1_score):
    """Test on correctness of f1_score calculation"""
    assert f1_score == cm(y_true, y_pred, fmt='labels', num_classes=2).f1_score()

def test_f1_calculation_multiclass():
    """Test on correctness of f1 score"""
    y_true, y_pred = np.array([2, 1]), np.array([[0.1, 0.1, 0.8], [0.1, 0.8, 0.1]])
    f_1 = 1.0
    assert f_1 == cm(y_true, y_pred, fmt='proba', axis=1, num_classes=3).f1_score()

# recall tests

@pytest.mark.parametrize('y_true, y_pred, tpr', [
    (np.array([1, 1, 0, 1]), np.array([0, 0, 1, 0]), 0.0),
    (np.array([1, 1, 0, 0]), np.array([0, 1, 1, 0]), 0.5),
    (np.array([1, 1, 0, 0]), np.array([1, 1, 1, 0]), 1.0)
])
def test_tpr_calculation(y_true, y_pred, tpr):
    """Test on correctness of true positive rate calculation"""
    assert tpr == cm(y_true, y_pred, fmt='labels', num_classes=2).true_positive_rate()

def test_tpr_calculation_multiclass():
    """Test on correctness of rrue positive rate calculation"""
    y_true, y_pred = np.array([2, 1]), np.array([[0.1, 0.1, 0.8], [0.1, 0.8, 0.1]])
    tpr = 1.0
    assert tpr == cm(y_true, y_pred, fmt='proba', axis=1, num_classes=3).true_positive_rate()

# false positive rate tests

@pytest.mark.parametrize('y_true, y_pred, fpr', [
    (np.array([0, 0, 0, 1]), np.array([0, 0, 0, 0]), 0.0),
    (np.array([1, 1, 0, 0]), np.array([0, 1, 1, 0]), 0.5),
    (np.array([0, 0, 0, 1]), np.array([1, 1, 1, 1]), 1.0)
])
def test_fpr_calculation(y_true, y_pred, fpr):
    """Test on correctness of false positive rate calculation"""
    assert fpr == cm(y_true, y_pred, fmt='labels', num_classes=2).false_positive_rate()

def test_fpr_calculation_multiclass():
    """Test on correctness of false positive rate calculation multiclass"""
    y_true, y_pred = np.array([2, 1]), np.array([[0.1, 0.1, 0.8], [0.1, 0.8, 0.1]])
    fpr = 0.0
    assert fpr == cm(y_true, y_pred, fmt='proba', axis=1, num_classes=3).false_positive_rate()
