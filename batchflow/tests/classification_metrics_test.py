"""File contains various tests for classification metrics in batchflow"""
import pytest
from batchflow.models.metrics import ClassificationMetrics as cm
import numpy as np

# define parameters

TEST_PARAMETERS = [
    (np.array([0, 1]), np.array(1)),
    (np.array(1), np.array([0, 1])),
    (np.array([[0, 1], [1, 0]]), np.array([0, 1])),
    (np.array([0, 1]), np.array([[0, 1], [1, 0]])),
    (np.array([[[0, 1], [1, 0]], [[1, 1], [0, 0]]]), np.array([[0, 1], [1, 0]])),
    (np.array([[0, 1], [1, 0]]), np.array([[[0, 1], [1, 0]], [[1, 1], [0, 0]]]))]

TEST_PARAM_MULTICLASS = [
    (np.array(2), np.array([[0.8, 0.1, 0.1], [0.1, 0.1, 0.8]])),
    (np.array([[[0, 1, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1]], [[1, 0, 0], [1, 1, 0], [1, 0, 1], [0, 0, 1]]]),
     np.array([[0.1, 0.8, 0.1], [0.1, 0.8, 0.1], [0.8, 0.1, 0.1], [0.8, 0.1, 0.1]]))]

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

# accuracy tests

@pytest.mark.parametrize('y_true,y_pred', TEST_PARAMETERS)
def test_diff_shapes_two_classes_acc(y_true, y_pred):
    """Testing different shape"""
    with pytest.raises(ValueError):
        cm(y_true, y_pred, fmt='labels', num_classes=2).accuracy()

@pytest.mark.parametrize('y_true,y_pred', TEST_PARAM_MULTICLASS)
def test_diff_shapes_multiclass_acc(y_true, y_pred):
    """Testing different shape"""
    with pytest.raises(ValueError):
        cm(y_true, y_pred, fmt='proba', axis=1, num_classes=3).accuracy()

def test_single_value_two_class_acc():
    """Test on accuracy single value in case of two class classification"""
    y_true, y_pred = np.random.choice([0, 1], size=(5,)), np.random.choice([0, 1], size=(5,))
    acc = cm(y_true, y_pred, fmt='labels', num_classes=2).accuracy()
    assert isinstance(acc, np.floating)

def test_vector_multiclass_acc():
    """Test on accuracy multiclass"""
    y_true = np.array([2, 1])
    y_pred = np.array([[0.1, 0.1, 0.8], [0.1, 0.8, 0.1]])
    acc = cm(y_true, y_pred, fmt='proba', axis=1, num_classes=3).accuracy()
    assert isinstance(acc, np.floating)

def test_vector_batches_two_class_acc():
    """Test on accuracy vector with batch shape if input is a multidimensional array"""
    y_true = np.array([[[1, 1], [0, 1]], [[0, 1], [1, 1]], [[1, 0], [1, 1]]])
    y_pred = np.array([[[0, 1], [1, 1]], [[1, 0], [0, 0]], [[0, 0], [0, 1]]])
    shape = cm(y_true, y_pred, fmt='labels', num_classes=2).accuracy().shape
    assert shape == (3,)

def test_vector_batches_multiclass_acc():
    """Test on accuracy vector with batch shape if input is a multidimensional array multiclass"""
    y_true = np.array([[[0, 1, 0], [1, 0, 0]], [[1, 0, 0], [0, 0, 1]]])
    y_pred = np.array([[[0.1, 0.8, 0.1], [0.8, 0.1, 0.1]], [[0.8, 0.1, 0.1], [0.1, 0.1, 0.8]]])
    shape = cm(y_true, y_pred, fmt='proba', axis=1, num_classes=3).accuracy().shape
    assert shape == (2,)

def test_axis_for_multiclass_acc():
    """Test on axis=None"""
    y_true, y_pred = np.array([2, 1]), np.array([[0.1, 0.1, 0.8], [0.1, 0.8, 0.1]])
    with pytest.raises(ValueError):
        cm(y_true, y_pred, fmt='proba', axis=None, num_classes=3).accuracy()

def test_accuracy_calculation():
    """Test on correctness of accuracy calculation"""
    y_true, y_pred = np.array([1, 1, 0, 1, 0, 0]), np.array([1, 1, 0, 0, 1, 1])
    accuracy = 0.5
    assert accuracy == cm(y_true, y_pred, fmt='labels', num_classes=2).accuracy()

def test_accuracy_calculation_multiclass():
    """Test on correctness of accuracy calculation"""
    y_true, y_pred = np.array([2, 1]), np.array([[0.1, 0.1, 0.8], [0.1, 0.8, 0.1]])
    accuracy = 1.0
    assert accuracy == cm(y_true, y_pred, fmt='proba', axis=1, num_classes=3).accuracy()

# f1 score tests

@pytest.mark.parametrize('y_true,y_pred', TEST_PARAMETERS)
def test_diff_shapes_two_classes_f1(y_true, y_pred):
    """Testing different shape"""
    with pytest.raises(ValueError):
        cm(y_true, y_pred, fmt='labels', num_classes=2).f1_score()

@pytest.mark.parametrize('y_true,y_pred', TEST_PARAM_MULTICLASS)
def test_diff_shapes_multiclass_f1(y_true, y_pred):
    """Testing different shape"""
    with pytest.raises(ValueError):
        cm(y_true, y_pred, fmt='proba', axis=1, num_classes=3).f1_score()

def test_single_value_two_class_f1():
    """Test on accuracy single value in case of two class classification"""
    y_true, y_pred = np.random.choice([0, 1], size=(5,)), np.random.choice([0, 1], size=(5,))
    shape = cm(y_true, y_pred, fmt='labels', num_classes=2).f1_score().shape
    assert shape == (1, 1)

def test_vector_multiclass_f1():
    """Test on f1 score shape with multiclass"""
    y_true, y_pred = np.array([2, 1]), np.array([[0.1, 0.1, 0.8], [0.1, 0.8, 0.1]])
    shape = cm(y_true, y_pred, fmt='proba', axis=1, num_classes=3).f1_score().shape
    assert shape == (1, 1)

def test_vector_batches_two_class_f1():
    """Test on f1 shape with bacth shape if input is a multidimensional array"""
    y_true = np.array([[[1, 1], [0, 1]], [[0, 1], [1, 1]], [[1, 0], [1, 1]]])
    y_pred = np.array([[[0, 1], [1, 1]], [[1, 0], [0, 0]], [[0, 0], [0, 1]]])
    shape = cm(y_true, y_pred, fmt='labels', num_classes=2).f1_score().shape
    assert shape == (3, 1)

def test_vector_batches_multiclass_f1():
    """Test on f1 shape with bacth shape multiclass"""
    y_true = np.array([[[0, 1, 0], [1, 0, 0]], [[1, 0, 0], [0, 0, 1]]])
    y_pred = np.array([[[0.1, 0.8, 0.1], [0.8, 0.1, 0.1]], [[0.8, 0.1, 0.1], [0.1, 0.1, 0.8]]])
    shape = cm(y_true, y_pred, fmt='proba', axis=1, num_classes=3).f1_score().shape
    assert shape == (2, 1)

def test_axis_for_multiclass_f1():
    """Test on axis=None"""
    y_true, y_pred = np.array([2, 1]), np.array([[0.1, 0.1, 0.8], [0.1, 0.8, 0.1]])
    with pytest.raises(ValueError):
        cm(y_true, y_pred, fmt='proba', axis=None, num_classes=3).f1_score()

def test_f1_calculation():
    """Test on correctness of f1_score calculation"""
    y_true, y_pred = np.array([1, 1, 0, 0]), np.array([0, 1, 1, 0])
    f_1 = 0.5
    assert f_1 == cm(y_true, y_pred, fmt='labels', num_classes=2).f1_score()

def test_f1_calculation_multiclass():
    """Test on correctness of f1 score"""
    y_true, y_pred = np.array([2, 1]), np.array([[0.1, 0.1, 0.8], [0.1, 0.8, 0.1]])
    f_1 = 1.0
    assert f_1 == cm(y_true, y_pred, fmt='proba', axis=1, num_classes=3).f1_score()

# recall tests

@pytest.mark.parametrize('y_true,y_pred', TEST_PARAMETERS)
def test_diff_shapes_two_classes_tpr(y_true, y_pred):
    """Testing different shape"""
    with pytest.raises(ValueError):
        cm(y_true, y_pred, fmt='labels', num_classes=2).true_positive_rate()

@pytest.mark.parametrize('y_true,y_pred', TEST_PARAM_MULTICLASS)
def test_diff_shapes_multiclass_tpr(y_true, y_pred):
    """Testing different shape"""
    with pytest.raises(ValueError):
        cm(y_true, y_pred, fmt='proba', axis=1, num_classes=3).true_positive_rate()

def test_single_value_two_class_tpr():
    """Test on tpr single value in case of two class classification"""
    y_true, y_pred = np.random.choice([0, 1], size=(5,)), np.random.choice([0, 1], size=(5,))
    shape = cm(y_true, y_pred, fmt='labels', num_classes=2).true_positive_rate().shape
    assert shape == (1, 1)

def test_vector_multiclass_tpr():
    """Test on tpr shape multiclass"""
    y_true = np.array([2, 1])
    y_pred = np.array([[0.1, 0.1, 0.8], [0.1, 0.8, 0.1]])
    shape = cm(y_true, y_pred, fmt='proba', axis=1, num_classes=3).true_positive_rate().shape
    assert shape == (1, 1)

def test_vector_batches_two_class_tpr():
    """Test on tpr shape with batch if input is a multidimensional array"""
    y_true = np.array([[[1, 1], [0, 1]], [[0, 1], [1, 1]], [[1, 0], [1, 1]]])
    y_pred = np.array([[[0, 1], [1, 1]], [[1, 0], [0, 0]], [[0, 0], [0, 1]]])
    shape = cm(y_true, y_pred, fmt='labels', num_classes=2).true_positive_rate().shape
    assert shape == (3, 1)

def test_vector_batches_multiclass_tpr():
    """Test on tpr shape with batch multiclass"""
    y_true = np.array([[[0, 1, 0], [1, 0, 0]], [[1, 0, 0], [0, 0, 1]]])
    y_pred = np.array([[[0.1, 0.8, 0.1], [0.8, 0.1, 0.1]], [[0.8, 0.1, 0.1], [0.1, 0.1, 0.8]]])
    shape = cm(y_true, y_pred, fmt='proba', axis=1, num_classes=3).true_positive_rate().shape
    assert shape == (2, 1)

def test_axis_for_multiclass_tpr():
    """Test on axis=None"""
    y_true, y_pred = np.array([2, 1]), np.array([[0.1, 0.1, 0.8], [0.1, 0.8, 0.1]])
    with pytest.raises(ValueError):
        cm(y_true, y_pred, fmt='proba', axis=None, num_classes=3).true_positive_rate()

def test_tpr_calculation():
    """Test on correctness of true positive rate calculation"""
    y_true, y_pred = np.array([1, 1, 0, 0]), np.array([0, 1, 1, 0])
    tpr = 0.5
    assert tpr == cm(y_true, y_pred, fmt='labels', num_classes=2).true_positive_rate()

def test_tpr_calculation_multiclass():
    """Test on correctness of rrue positive rate calculation"""
    y_true, y_pred = np.array([2, 1]), np.array([[0.1, 0.1, 0.8], [0.1, 0.8, 0.1]])
    tpr = 1.0
    assert tpr == cm(y_true, y_pred, fmt='proba', axis=1, num_classes=3).true_positive_rate()

# false positive rate tests

@pytest.mark.parametrize('y_true,y_pred', TEST_PARAMETERS)
def test_diff_shapes_two_classes_fpr(y_true, y_pred):
    """Testing different shape"""
    with pytest.raises(ValueError):
        cm(y_true, y_pred, fmt='labels', num_classes=2).false_positive_rate()

@pytest.mark.parametrize('y_true,y_pred', TEST_PARAM_MULTICLASS)
def test_diff_shapes_multiclass_fpr(y_true, y_pred):
    """Testing different shape"""
    with pytest.raises(ValueError):
        cm(y_true, y_pred, fmt='proba', axis=1, num_classes=3).false_positive_rate()

def test_single_value_two_class_fpr():
    """Test on fpr shape in case of two class classification"""
    y_true, y_pred = np.array([0, 1, 0, 1]), np.array([0, 1, 1, 1])
    shape = cm(y_true, y_pred, fmt='labels', num_classes=2).false_positive_rate().shape
    assert shape == (1, 1)

def test_vector_multiclass_fpr():
    """Test on fpr shape multiclass"""
    y_true = np.array([2, 1])
    y_pred = np.array([[0.1, 0.1, 0.8], [0.1, 0.8, 0.1]])
    shape = cm(y_true, y_pred, fmt='proba', axis=1, num_classes=3).false_positive_rate().shape
    assert shape == (1, 1)

def test_vector_batches_two_class_fpr():
    """Test on fpr shape with batch if input is a multidimensional array"""
    y_true = np.array([[[1, 1], [0, 1]], [[0, 1], [1, 1]], [[1, 0], [1, 1]]])
    y_pred = np.array([[[0, 1], [1, 1]], [[1, 0], [0, 0]], [[0, 0], [0, 1]]])
    shape = cm(y_true, y_pred, fmt='labels', num_classes=2).false_positive_rate().shape
    assert shape == (3, 1)

def test_vector_batches_multiclass_fpr():
    """Test on fpr shape with batch multiclass"""
    y_true = np.array([[[0, 1, 0], [1, 0, 0]], [[1, 0, 0], [0, 0, 1]]])
    y_pred = np.array([[[0.1, 0.8, 0.1], [0.8, 0.1, 0.1]], [[0.8, 0.1, 0.1], [0.1, 0.1, 0.8]]])
    shape = cm(y_true, y_pred, fmt='proba', axis=1, num_classes=3).false_positive_rate().shape
    assert shape == (2, 1)

def test_axis_for_multiclass_fpr():
    """Test on axis=None"""
    y_true, y_pred = np.array([2, 1]), np.array([[0.1, 0.1, 0.8], [0.1, 0.8, 0.1]])
    with pytest.raises(ValueError):
        cm(y_true, y_pred, fmt='proba', axis=None, num_classes=3).false_positive_rate()

def test_fpr_calculation():
    """Test on correctness of false positive rate calculation"""
    y_true, y_pred = np.array([1, 1, 0, 0]), np.array([0, 1, 1, 0])
    fpr = 0.5
    assert fpr == cm(y_true, y_pred, fmt='labels', num_classes=2).false_positive_rate()

def test_fpr_calculation_multiclass():
    """Test on correctness of false positive rate calculation multiclass"""
    y_true, y_pred = np.array([2, 1]), np.array([[0.1, 0.1, 0.8], [0.1, 0.8, 0.1]])
    fpr = 0.0
    assert fpr == cm(y_true, y_pred, fmt='proba', axis=1, num_classes=3).false_positive_rate()
