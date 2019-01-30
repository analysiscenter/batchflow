import sys
import pytest
import numpy as np

sys.path.append('..')
from batchflow.dsindex import DatasetIndex


class ChildSet(DatasetIndex):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

SIZE = 10

@pytest.fixture
def dsindex_int():
	return DatasetIndex(SIZE)

@pytest.fixture
def dsindex_list():
	return DatasetIndex(list(range(2*SIZE, 4*SIZE)))

@pytest.fixture
def dsindex_callable():
	np.random.seed(0)
	return DatasetIndex(lambda: np.random.random(SIZE))

@pytest.fixture
def dsindex_big():
	return DatasetIndex(SIZE**3)



class TestBaseset:

	def test_baseset_len(self, dsindex_int, dsindex_list, dsindex_callable):
		dsindex_rec = DatasetIndex(dsindex_list)
		assert len(dsindex_int) == SIZE
		assert len(dsindex_list) == 2*SIZE
		assert len(dsindex_callable) == SIZE
		assert len(dsindex_rec) == 2*SIZE

	def test_baseset_calc_split_shares(self, dsindex_big):
		with pytest.raises(ValueError):
			dsindex_big.calc_split(shares=[0.5, 0.5, 0.5])
		with pytest.raises(ValueError):
			dsindex_big.calc_split(shares=[0.5, 0.5, 0.5, 0.5]) 
		with pytest.raises(ValueError):
			DatasetIndex(2).calc_split(shares=[0.5, 0.5, 0.5])

	def test_baseset_calc_split_correctness_1(self, dsindex_big):
		assert sum(dsindex_big.calc_split()) == SIZE**3

	@pytest.mark.skip(reason='line 94 of batchflow/base.py')
	def test_baseset_calc_split_correctness_2(self, dsindex_big):
		left = dsindex_big.calc_split(shares=[0.5, 0.5])
		right = dsindex_big.calc_split(shares=[0.5, 0.5, 0.0])
		assert left == right
		
	def test_baseset_calc_split_correctness_3(self, dsindex_big):
		left = dsindex_big.calc_split(shares=[0.5, 0.5])
		right = (0.5*SIZE**3, 0.5*SIZE**3, 0)
		assert left == right


class TestDatasetIndex:


	def test_build_index_int(self, dsindex_int):
		assert (dsindex_int.index == np.arange(SIZE)).all()
		
	def test_build_index_list(self, dsindex_list):
		assert (dsindex_list.index == list(range(2*SIZE, 4*SIZE))).all()
		
	def test_build_index_callable(self, dsindex_callable):
		np.random.seed(0)
		assert (dsindex_callable.index == np.random.random(SIZE)).all()

	def test_build_index_dsindex(self, dsindex_list):
		dsindex_rec = DatasetIndex(dsindex_list)
		assert (dsindex_rec.index == dsindex_list.index).all()
		
	def test_build_index_empty(self):
		with pytest.raises(ValueError):
			dsindex_bad = DatasetIndex([])

	def test_build_index_multidimensional(self):
		with pytest.raises(TypeError):
			dsindex_bad = DatasetIndex(np.random.random(size=(SIZE, SIZE)))


	def test_get_pos_slice(self, dsindex_list):
		assert dsindex_list.get_pos(slice(2*SIZE, 3*SIZE, 1)) == slice(0, SIZE, 1)
		
	def test_get_pos_str(self):
		dsindex_str = DatasetIndex(['a', 'b', 'c', 'd', 'e'])
		assert dsindex_str.get_pos('a') == 0
		
	def test_get_pos_str_iterable(self):
		dsindex_str = DatasetIndex(['a', 'b', 'c', 'd', 'e'])
		assert set(dsindex_str.get_pos(['a', 'b'])) == set(np.array([0, 1]))
		assert (dsindex_str.get_pos(['a', 'b']) == np.array([0, 1])).all()
		
	def test_get_pos_int(self, dsindex_int):
		assert dsindex_int.get_pos(SIZE-1) == SIZE-1
		
	def test_get_pos_iterable(self, dsindex_list):
		assert set(dsindex_list.get_pos(range(2*SIZE, 3*SIZE))) == set(range(0, SIZE))
		assert (dsindex_list.get_pos(range(2*SIZE, 3*SIZE)) == range(0, SIZE)).all() 




	def test_shuffle_bool_false(self, dsindex_list):
		left = dsindex_list._shuffle(shuffle=False) 
		right = np.arange(len(dsindex_list))
		assert (left == right).all()
		
	def test_shuffle_bool_true(self, dsindex_list):
		left = dsindex_list._shuffle(shuffle=True)
		assert (left != np.arange(len(dsindex_list))).any()
		assert set(left) == set(np.arange(len(dsindex_list)))
		
	def test_shuffle_int(self, dsindex_list):
		left = dsindex_list._shuffle(shuffle=SIZE)
		assert (left != np.arange(len(dsindex_list))).any()
		assert (left == dsindex_list._shuffle(shuffle=SIZE)).all()
		assert set(left) == set(np.arange(len(dsindex_list)))    

	def test_shuffle_randomstate(self, dsindex_list):
		left = dsindex_list._shuffle(shuffle=np.random.RandomState())
		assert (left != np.arange(len(dsindex_list))).any()
		left = dsindex_list._shuffle(shuffle=np.random.RandomState(SIZE))
		right = dsindex_list._shuffle(shuffle=np.random.RandomState(SIZE))
		assert (left == right).all()
		left = dsindex_list._shuffle(shuffle=np.random.RandomState())
		assert set(left) == set(np.arange(len(dsindex_list)))  
		
	def test_shuffle_cross(self, dsindex_list):
		left = dsindex_list._shuffle(shuffle=SIZE)
		right = dsindex_list._shuffle(shuffle=np.random.RandomState(SIZE))
		assert (left == right).all()

	@pytest.mark.skip(reason='permutes index, not order')    
	def test_shuffle_callable(self, dsindex_list):
		left = dsindex_list._shuffle(shuffle=np.random.permutation) 
		assert (left != np.arange(len(dsindex_list))).all()
		assert set(left) == set(np.arange(len(dsindex_list)))




