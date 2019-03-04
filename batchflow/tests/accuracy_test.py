"""importing pytest, numpy and metrics from batchflow"""
import pytest
from batchflow.models import metrics
import numpy as np

def test_different_shapes():
    """Testing different shape"""
    y_true, y_pred = np.random.choice([0, 1], size=(5,)), np.random.choice([0, 1], size=(4, 4))
    with pytest.raises(ValueError):
        metrics.ClassificationMetrics(y_true, y_pred,
                                      fmt='labels', num_classes=2).accuracy()

def test_single_value_two_class():
    """Test on accuracy single value in case of two class classification"""
    y_true, y_pred = np.random.choice([0, 1], size=(5,)), np.random.choice([0, 1], size=(5,))
    acc = metrics.ClassificationMetrics(y_true, y_pred,
                                        fmt='labels', num_classes=2).accuracy()
    assert isinstance(acc, np.floating)

def test_vector_batches_two_class():
    """Test on accuracy vector with bacth shape if input is a multidimensional array"""
    y_true = np.array([[[1, 1], [0, 1]], [[0, 1], [1, 1]], [[1, 0], [1, 1]]])
    y_pred = np.array([[[0, 1], [1, 1]], [[1, 0], [0, 0]], [[0, 0], [0, 1]]])
    assert metrics.ClassificationMetrics(y_true, y_pred,
                                         fmt='labels', num_classes=2).accuracy().shape[0] > 1

def test_accuracy_boundaries():
    """Test on accuracy single metrics boundaries"""
    y_true, y_pred = np.random.choice([0, 1], size=(5,)), np.random.choice([0, 1], size=(5,))
    acc = metrics.ClassificationMetrics(y_true, y_pred,
                                        fmt='labels', num_classes=2).accuracy()
    assert (acc >= 0) & (acc <= 1)

def test_confusion_matrix():
    """Test on correctness of confusion matrix calculation"""
    y_true, y_pred = np.array([1, 1, 0, 1, 0, 0]), np.array([0, 0, 1, 0, 0, 0])
    conf_matrix = np.array([[2, 3], [1, 0]])
    conf_matrix_calc = metrics.ClassificationMetrics(y_true, y_pred,                                #pylint:disable=protected-access
                                                     fmt='labels', num_classes=2)._confusion_matrix
    assert (conf_matrix_calc == conf_matrix).all()

def test_accuracy_calculation():
    """Test on correctness of accuracy calculation"""
    y_true, y_pred = np.array([1, 1, 0, 1, 0, 0]), np.array([0, 0, 1, 0, 0, 0])
    t_p, f_p, f_n, t_n = 0, 1, 3, 2
    accuracy = (t_p + t_n) / (t_p + t_n + f_p + f_n)
    assert accuracy == metrics.ClassificationMetrics(y_true, y_pred,
                                                     fmt='labels', num_classes=2).accuracy()

def test_f1_vs_accuracy():
    """Test on comparison f1 and accuracy values in case of inbalanced classes"""
    y_true, y_pred = np.array([0, 0, 0, 0, 1]), np.array([0, 0, 0, 0, 0])
    f1_score_result = metrics.ClassificationMetrics(y_true, y_pred,
                                                    fmt='labels', num_classes=2).f1_score()
    accuracy_result = metrics.ClassificationMetrics(y_true, y_pred,
                                                    fmt='labels', num_classes=2).accuracy()
    assert f1_score_result < accuracy_result
