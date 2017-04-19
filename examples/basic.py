# pylint: skip-file
import os
import sys
import numpy as np
import pandas as pd

sys.path.append("..")
from dataset import * # pylint: disable=wrong-import-

if __name__ == "__main__":
    # number of items in the dataset
    K = 10

    # Fill-in dataset with sample data
    def pd_data():
        ix = np.arange(K)
        data = np.arange(K * 3).reshape(K, -1)
        dsindex = DatasetIndex(ix)
        ds = Dataset(index=dsindex, batch_class=ArrayBatch)
        return ds, data.copy()

    # Create datasets
    ds_data, data = pd_data()

    print("Start iterating...")
    for i in range(K + 5):
        batch = ds_data.next_batch(1, shuffle=False, one_pass=True)
        print("batch", i, ":", batch.indices)
    print("End iterating")
