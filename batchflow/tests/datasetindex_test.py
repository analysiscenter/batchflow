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







