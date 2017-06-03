# pylint: skip-file
import os
import sys
import asyncio
import numpy as np
import pandas as pd
from numba import njit
import blosc
from time import time, sleep

sys.path.append("..")
from dataset import * # pylint: disable=wrong-import-


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

    def parallel_init(self, *args, **kwargs):
        r = self.indices.tolist()
        print("\n Init:", r)
        return r

    def parallel_post(self, results, not_done=None):
        print(" Post:", results)
        return self

    @action
    def action0(self, *args):
        print("      batch", self.indices, "   action 0", args)
        sleep(10)
        return self

    @staticmethod
    def _write_file(path, attr, data):
        with open(path, 'w' + attr) as file:
            file.write(data)

    @action
    def dump(self, *args):
        print("         dumping:", self.indices)
        fullname = './data/many/'+str(int(np.random.uniform(0, 1) * 123456789))
        packed_array = blosc.pack_array(self.data)
        self._write_file(fullname, 'b', packed_array)
        return self


if __name__ == "__main__":
    # number of items in the dataset
    K = 10000

    # Fill-in dataset with sample data
    def pd_data():
        ix = np.arange(K)  #.astype('str')
        data = np.arange(K * 3).reshape(K, -1)
        dsindex = DatasetIndex(ix)
        ds = Dataset(index=dsindex, batch_class=MyArrayBatch)
        return ds, data.copy()


    # Create datasets
    ds_data, data = pd_data()

    res = (ds_data.pipeline()
            .print("Start batch")
            .load(data)
            .action0()
            .dump()
            .print("End batch"))

    #res.run(4, shuffle=False)
    print("Start iterating...")
    t = time()
    res.run(10, shuffle=False, n_epochs=1, drop_last=True, prefetch=100, target='mpc')
    print("End:", time() - t)

#    i = 0
#    for batch_res in res.gen_batch(3, shuffle=False, n_epochs=1, prefetch=1, target='mpc'):
#        print('-------------------------------------------------')
#        print("====== Iteration ", i, "batch:", batch_res.indices)
#        i += 1
#    print("====== Stop iteration ===== ")

