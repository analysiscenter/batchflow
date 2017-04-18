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
def numba_fn(k, a1=0, a2=0, a3=0):
    print("   action numba", k, "started", a1, a2, a3)
    for i in range(k * 3000):
        x = np.random.normal(0, 1, size=10000)
    print("   action numba", k, "ended")
    return x

def mpc_fn(i, *args):
    print("   mpc func", i, args)
    return i


# Example of custome Batch class which defines some actions
class MyDataFrameBatch(DataFrameBatch):
    @action
    def print(self, text=None):
        if text is not None:
            print(text)
        print(self.data)
        return self

    def parallel_init(self, *args, **kwargs):
        r = self.indices.tolist()
        print("Parallel:", r)
        return r

    def parallel_post(self, results, not_done=None):
        print("Post:", results)
        return self


    @action
    @inbatch_parallel(init="parallel_init", target='mpc') #, post="parallel_post")
    def action1(self, *args):
        print("   action 1", args)
        return mpc_fn

    def action_n_init(self, *args, **kwargs):
        r = self.indices.astype('int').tolist()
        print("Parallel:", r)
        return r

    @action
    @inbatch_parallel(init="action_n_init", target="nogil")
    def action_n(self, *args, **kwargs):
        return numba_fn

    @action
    @inbatch_parallel(init="parallel_init", post="parallel_post", target='async')
    async def action2(self, i, *args):
        print("   action 2", i, "started", args)
        await asyncio.sleep(1)
        print("   action 2", i, "ended")
        return i

    @action
    def add(self, inc):
        self.data += inc
        return self


if __name__ == "__main__":
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
            .action2("async")
            .action_n(123)
            .action1(17)
            .print("End batch"))

    res.run(4, shuffle=False)
