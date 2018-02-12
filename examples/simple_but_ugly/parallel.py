# pylint: skip-file
import os
import sys
import asyncio
from functools import partial
import numpy as np
from numba import njit

sys.path.append("../..")
from dataset import * # pylint: disable=wrong-import-


@njit(nogil=True)
def numba_fn(k, a1=0, a2=0, a3=0):
    print("   action numba", k, "started", a1, a2, a3)
    if k > 8:
        print("         fail:", k)
        y = 12 / np.log(1)
    for i in range((k + 1) * 3000):
        x = np.random.normal(0, 1, size=10000)
    print("   action numba", k, "ended")
    return x

def mpc_fn(i, arg2):
    print("   mpc func", i, arg2)
    if i > 8:
        y = 12 / np.log(1)
    else:
        y = i
    return y


# Example of custome Batch class which defines some actions
class MyBatch(Batch):
    @action
    def print(self, text=None):
        if text is not None:
            print("\n=====", text, "=====")
        print(self.data)
        return self

    def parallel_post(self, results, *args, **kwargs):
        print("Post:")
        print("   any failed?", any_action_failed(results))
        print("  ", results)
        return self

    @action
    @inbatch_parallel(init="indices", post="parallel_post", target='mpc')
    def action_p(self, *args, **kwargs):
        print("   action mpc", args)
        return mpc_fn

    @action
    @inbatch_parallel(init="indices")
    def action_t(self, ix, value, **kwargs):
        print("   action threads", ix, value, kwargs)

    @action
    @inbatch_parallel(init="items")
    def action_i(self, item, *args, **kwargs):
        print("   action items", type(item), item)
        return self

    @action
    @inbatch_parallel(init="indices", post="parallel_post", target="threads")
    def action_n(self, *args, **kwargs):
        return numba_fn(*args, **kwargs)

    @action
    @inbatch_parallel(init="indices", post="parallel_post", target='async')
    async def action_a(self, ix, *args):
        print("   action a", ix, "started", args)
        if ix == '2':
            print("   action 2", ix, "failed")
            x = 12 / 0
        else:
            await asyncio.sleep(1)
        print("   action 2", ix, "ended")
        return ix

    @action
    def add(self, inc):
        self.data += inc
        return self

    @action
    @inbatch_parallel(init="items")
    @mjit
    def act(self, data):
        data[:] = np.log(data ** 2)


if __name__ == "__main__":
    # number of items in the dataset
    K = 10

    # Fill-in dataset with sample data
    def gen_data():
        ix = np.arange(K)
        data = np.arange(K * 3).reshape(K, -1).astype('float32')
        ds = Dataset(index=ix, batch_class=MyBatch)
        return ds, data


    # Create datasets
    ds_data, data = gen_data()

    res = (ds_data.pipeline()
            .load(data)
            .print("Start batch")
            #.action_p(S('uniform', 10, 15))
            #.action_a("async", P(R('poisson', 5.5)))
            #.action_i(P(R([500, 600])))
            #.action_t(P(R('normal', 10, 2)), target='f')
            #.action1(arg2=14)
            .act()
            .print("End batch", F(lambda b: b.data[0])))

    res.run(4, shuffle=False, n_epochs=1)
