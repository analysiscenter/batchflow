# pylint: skip-file
import os
import sys
import numpy as np
import pandas as pd

sys.path.append("..")
from dataset import * # pylint: disable=wrong-import-

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

    def action1_post(self, results, not_done):
        print("Post:")
        for item in results:
            print("  ", item)
        return self

    @action
    @within_parallel(init="action1_init", post="action1_post")
    def action1(self, i):
        print("   action 1", i)
        return self

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
        .action1()
        .print("End batch"))

res.run(4, shuffle=False)
