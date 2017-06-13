# pylint: skip-file
import os
import sys
import numpy as np
from numba import njit
import PIL.Image
import scipy.ndimage
from time import time

sys.path.append("..")
from dataset import DatasetIndex, Dataset, ImagesBatch, action, inbatch_parallel, any_action_failed


# Example of custom Batch class which defines some actions
class MyImages(ImagesBatch):
    @action
    @inbatch_parallel(init='indices', post='assemble')
    def load(self, ix, src, fmt=None):
        if fmt == 'PIL':
            return PIL.Image.fromarray(src[ix])
        else:
            return src[ix]


if __name__ == "__main__":
    # number of items in the dataset
    K = 1000
    S = 128

    # Fill-in dataset with sample data
    def gen_data():
        ix = np.arange(K)
        data = np.random.randint(0, 255, size=K*S*S).reshape(K, S, S)
        dsindex = DatasetIndex(ix)
        ds = Dataset(index=dsindex, batch_class=MyImages)
        return ds, data


    # Create datasets
    print("Generating...")
    ds_data, data = gen_data()

    res = ds_data.p.load(data).convert_to_PIL('images').resize((384, 384))
    #res = ds_data.p.load(data).resize((384, 384), method='PIL')

    print("Start...")
    t = time()
    res.run(K//10, n_epochs=1)
    print("End", time() - t)
