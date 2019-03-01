import pytest
from batchflow.models import metrics
import numpy as np
	
def test_different_shapes():
		
	y_true, y_pred = np.random.choice( [0, 1], size = (5,) ), np.random.choice( [0, 1], size = (4,4) ) 
		
	with pytest.raises(ValueError):
			
		metrics.ClassificationMetrics(y_true, y_pred, fmt = 'labels', num_classes = 2 ).accuracy()
		
def test_single_value_two_class_classification():

	y_true, y_pred = np.random.choice( [0, 1], size = (5,) ), np.random.choice( [0, 1], size = (5,) )

	assert isinstance( metrics.ClassificationMetrics(y_true, y_pred, fmt = 'labels', num_classes = 2).accuracy(),
           np.floating )
		   
def test_vector_batches_two_class_classification():

	y_true, y_pred = np.array( [[[1,1],[0,1]],[[0,1],[1,1]],[[1,0],[1,1]]] ), np.array( [[[0,1],[1,1]],[[1,0],[0,0]],[[0,0],[0,1]]])
	
	assert metrics.ClassificationMetrics(y_true, y_pred, fmt = 'labels', num_classes = 2).accuracy().shape[0] > 1
           	
	
def test_accuracy_between_zero_and_one():
	
	y_true, y_pred = np.random.choice( [0, 1], size = (5,) ), np.random.choice( [0, 1], size = (5,) )
	
	assert  ( 0 <= metrics.ClassificationMetrics(y_true, y_pred, fmt = 'labels', num_classes = 2).accuracy()  )  &  ( metrics.ClassificationMetrics(y_true, y_pred, fmt='labels', num_classes=2).accuracy() <= 1 )
	
def test_confusion_matrix():

	y_true, y_pred = np.array ( [1, 1, 0, 1, 0, 0] ), np.array( [0, 0, 1, 0, 0, 0] )
	
	conf_matrix = np.array( [ [2, 3], [1, 0] ] )
	
	assert ( metrics.ClassificationMetrics(y_true, y_pred,fmt='labels',num_classes=2)._confusion_matrix == conf_matrix ).all()
	
def test_accuracy_calculation():

	y_true, y_pred = np.array ( [1, 1, 0, 1, 0, 0] ), np.array( [0, 0, 1, 0, 0, 0] )
	
	TP, FP, FN, TN = 0, 1, 3, 2
	
	accuracy = (TP + TN) / ( TP + TN + FP + FN ) 
	
	assert accuracy == metrics.ClassificationMetrics(y_true, y_pred,fmt='labels',num_classes=2).accuracy()
	
def test_f1_vs_accuracy_imbalanced_classes():

		y_true, y_pred = np.array( [0,0,0,0,1] ), np.array( [0, 0, 0, 0, 0] )
    	
		f1_score_result = metrics.ClassificationMetrics(y_true,y_pred, fmt = 'labels', num_classes = 2).f1_score()
		
		accuracy_result = metrics.ClassificationMetrics(y_true,y_pred, fmt = 'labels', num_classes = 2).accuracy()
		
		assert f1_score_result < accuracy_result
	
