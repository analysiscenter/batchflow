# pylint: skip-file
import os
import sys
import numpy as np
import pandas as pd

sys.path.append("../..")
from dataset import Dataset, DatasetIndex, ArrayBatch


# Make a dataset with sample data
def gen_data(num_items):
    ix = np.arange(num_items)
    data = np.arange(num_items * 3).reshape(num_items, -1)
    dsindex = DatasetIndex(ix)
    ds = Dataset(index=dsindex, batch_class=ArrayBatch)
    return ds, data


if __name__ == "__main__":
    # number of items in the dataset
    NUM_ITEMS = 10
    BATCH_SIZE = 3

    # Create datasets
    ds_data, data = gen_data(NUM_ITEMS)

    print("Start iterating...")
    i = 0
    for batch in ds_data.gen_batch(BATCH_SIZE, n_epochs=1):
        i += 1
        print("batch", i, " contains items", batch.indices)
    print("End iterating")

    print()
    print("And now with drop_last=True")
    print("Start iterating...")
    i = 0
    for batch in ds_data.gen_batch(BATCH_SIZE, n_epochs=1, drop_last=True):
        i += 1
        print("batch", i, " contains items", batch.indices)
    print("End iterating")

    print()
    print("And one more time, but with next_batch(...) and too many iterations, so will get a StopIteration")
    print("Start iterating...")
    for i in range(NUM_ITEMS * 3):
        batch = ds_data.next_batch(BATCH_SIZE, n_epochs=2)
        print("batch", i + 1, "contains items", batch.indices)
    print("End iterating")
