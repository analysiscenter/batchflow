# pylint: skip-file
import os
import sys
from time import time
import numpy as np

sys.path.append('..')
from dataset import *


# Example of custome Batch class which defines some actions
class MyArrayBatch(ArrayBatch):
    def __init__(self, index, *args, **kwargs):
        super().__init__(index)

    @action
    def print(self, text=None):
        if text is not None:
            print("\n====", text, self.indices, "======\n")
        print(self.data)
        return self

    def parallel_post(self, results, not_done=None):
        print(" Post:", results)
        return self

    @action
    def action0(self, *args):
        print("   batch", self.indices, "   action 0", args)
        return self

    @model()
    def sm(self):
        return [1,2,3]

    @action
    def action_m(self, model, arg=5):
        print("   batch", self.indices, "   action m", model, arg)
        return self

    @action
    @inbatch_parallel(init="indices") #, post="parallel_post")
    def action1(self, i, *args):
        print("   batch", self.indices, "   action 1", i, args)
        return i


# number of items in the dataset
K = 100


# Fill-in dataset with sample data
def pd_data():
    ix = np.arange(K)
    data = np.arange(K * 3).reshape(K, -1).astype("float")
    dsindex = DatasetIndex(ix)
    ds = Dataset(index=dsindex, batch_class=MyArrayBatch)
    return ds, data.copy()


# Create datasets
ds_data, data = pd_data()


# Create pipeline
res = (ds_data.pipeline()
        .load(data)
        .action0()
        .action_m(3)
        .action1()
)


print("Start iterating...")
t = time()
t1 = t
for batch in res.gen_batch(2, shuffle=False, n_epochs=1, drop_last=True, prefetch=Q*2, tf_session=sess, target='threads'):
    print("Batch", batch.indices, "is ready in", time() - t1)
    t1 = time()
    pass

print("Stop iterating:", time() - t)
