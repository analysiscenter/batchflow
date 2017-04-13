# pylint: skip-file
import os
import sys
import asyncio
import numpy as np
import pandas as pd
from numba import njit

sys.path.append("..")
from dataset import * # pylint: disable=wrong-import-


@njit(nogil=True)
def numba_fn(k):
    print("Start:", k)
    for i in range(k * 10000):
        x = np.random.normal(0, 1, size=10000)
    print("End:", k)
    return x


# Example of custome Batch class which defines some actions
class MyDataFrameBatch(DataFrameBatch):
    @action
    def print(self, text=None):
        if text is not None:
            print(text)
        print(self.data)
        return self

    def action1_init(self, *args, **kwargs):
        print("Parallel:")
        return np.arange(3).reshape(3, -1).tolist()

    def action1_post(self, results, not_done=None):
        print("Post:", results)
        return self

    @action
    @inbatch_parallel(init="action1_init") #, post="action1_post")
    def action1(self, i):
        print("   action 1", i)
        return i

    @action
    @inbatch_parallel(init="action1_init", target="nogil")
    def action_n(self):
        return numba_fn

    def action1(self, i):
        print("   action 1", i)        
        return i

    @action
    @inbatch_parallel(init="action1_init", post="action1_post", target='async')
    async def action2(self, i):
        print("   action 2", i, "started")
        await asyncio.sleep(1)
        print("   action 2", i, "ended")
        return i

    @action
    def add(self, inc):
        self.data += inc
        return self


# number of items in the dataset
K = 10

# Fill-in dataset with sample data
def pd_data():
    ix = np.arange(K).astype('str')
    data = pd.DataFrame(np.arange(K * 3).reshape(K, -1), index=ix)
    dsindex = DatasetIndex(data.index)
    ds = Dataset(index=dsindex, batch_class=MyDataFrameBatch)
    return ds, data.copy()


# Create datasets
ds_data, data = pd_data()

res = (ds_data.pipeline()
        .load(data)
        .print("\nStart batch")
        .action_n()
        .print("End batch"))

res.run(4, shuffle=False)
