"""File contains various tests for classification metrics in batchflow"""
import pytest
from batchflow.models.metrics import ClassificationMetrics as cm
import numpy as np

@pytest.mark.parametrize('y_true,y_pred', [
    (np.array([0, 1]), np.array(1)),
    (np.array([[0, 1], [1, 0]]), np.array([0, 1])),
    (np.array([[[0, 1], [1, 0]], [[1, 1], [0, 0]]]), np.array([[0, 1], [1, 0]])),
])
def test_diff_shapes_two_classes(y_true, y_pred):
    """Testing different shape"""
    with pytest.raises(ValueError):
        cm(y_true, y_pred, fmt='labels', num_classes=2).accuracy()
# может быть еще стоит передавать в функцию метрику

def test_single_value_two_class():
    """Test on accuracy single value in case of two class classification"""
    y_true, y_pred = np.random.choice([0, 1], size=(5,)), np.random.choice([0, 1], size=(5,))
    acc = cm(y_true, y_pred, fmt='labels', num_classes=2).accuracy()
    assert isinstance(acc, np.floating)

def test_axis_for_multiclass():
    """Test on axis=None"""
    y_true, y_pred = np.array([2, 3]), np.array([[0.1, 0.7, 0.9], [0.1, 0.9, 0.4]])
    with pytest.raises(ValueError):
        cm(y_true, y_pred, fmt='proba', axis=None, num_classes=3).accuracy()
#тут может передать proba и logit через parametrize

def test_vector_batches_two_class():
    """Test on accuracy vector with bacth shape if input is a multidimensional array"""
    y_true = np.array([[[1, 1], [0, 1]], [[0, 1], [1, 1]], [[1, 0], [1, 1]]])
    y_pred = np.array([[[0, 1], [1, 1]], [[1, 0], [0, 0]], [[0, 0], [0, 1]]])
    assert cm(y_true, y_pred, fmt='labels', num_classes=2).accuracy().shape[0] > 1

def test_accuracy_boundaries():
    """Test on accuracy single metrics boundaries"""
    y_true, y_pred = np.random.choice([0, 1], size=(5,)), np.random.choice([0, 1], size=(5,))
    acc = cm(y_true, y_pred, fmt='labels', num_classes=2).accuracy()
    assert (acc >= 0) & (acc <= 1)

def test_confusion_matrix():
    """Test on correctness of confusion matrix calculation"""
    y_true, y_pred = np.array([1, 1, 0, 1, 0, 0]), np.array([0, 0, 1, 0, 0, 0])
    conf_matrix = np.array([[2, 3], [1, 0]])
    conf_matrix_calc = cm(y_true, y_pred, fmt='labels', num_classes=2)._confusion_matrix  #pylint:disable=protected-access
    assert (conf_matrix_calc == conf_matrix).all()

def test_accuracy_calculation():
    """Test on correctness of accuracy calculation"""
    y_true, y_pred = np.array([1, 1, 0, 1, 0, 0]), np.array([1, 1, 0, 0, 1, 1])
    accuracy = 0.5
    assert accuracy == cm(y_true, y_pred, fmt='labels', num_classes=2).accuracy()

@pytest.mark.parametrize('y_true,y_pred', [
    (np.array([0, 1]), np.array(1)),
    (np.array([[0, 1], [1, 0]]), np.array([0, 1])),
    (np.array([[[0, 1], [1, 0]], [[1, 1], [0, 0]]]), np.array([[0, 1], [1, 0]])),
])
def test_diff_shapes_two_classes_f1(y_true, y_pred):
    """Testing different shape"""
    with pytest.raises(ValueError):
        cm(y_true, y_pred, fmt='labels', num_classes=2).f1_score()

def test_f1_calculation():
    """Test on correctness of f1_score calculation"""
    y_true, y_pred = np.array([1, 1, 0, 0]), np.array([0, 1, 1, 0])
    f_1 = 0.5
    assert f_1 == cm(y_true, y_pred, fmt='labels', num_classes=2).f1_score()

@pytest.mark.parametrize('y_true,y_pred', [
    (np.array([0, 1]), np.array(1)),
    (np.array([[0, 1], [1, 0]]), np.array([0, 1])),
    (np.array([[[0, 1], [1, 0]], [[1, 1], [0, 0]]]), np.array([[0, 1], [1, 0]])),
])
def test_diff_shapes_two_classes_tpr(y_true, y_pred):
    """Testing different shape"""
    with pytest.raises(ValueError):
        cm(y_true, y_pred, fmt='labels', num_classes=2).true_positive_rate()

def test_tpr_calculation():
    """Test on correctness of true positive rate calculation"""
    y_true, y_pred = np.array([1, 1, 0, 0]), np.array([0, 1, 1, 0])
    tpr = 0.5
    assert tpr == cm(y_true, y_pred, fmt='labels', num_classes=2).true_positive_rate()

@pytest.mark.parametrize('y_true,y_pred', [
    (np.array([0, 1]), np.array(1)),
    (np.array([[0, 1], [1, 0]]), np.array([0, 1])),
    (np.array([[[0, 1], [1, 0]], [[1, 1], [0, 0]]]), np.array([[0, 1], [1, 0]])),
])
def test_diff_shapes_two_classes_fpr(y_true, y_pred):
    """Testing different shape"""
    with pytest.raises(ValueError):
        cm(y_true, y_pred, fmt='labels', num_classes=2).false_positive_rate()

def test_fpr_calculation():
    """Test on correctness of false positive rate calculation"""
    y_true, y_pred = np.array([1, 1, 0, 0]), np.array([0, 1, 1, 0])
    fpr = 0.5
    assert fpr == cm(y_true, y_pred, fmt='labels', num_classes=2).false_positive_rate()
