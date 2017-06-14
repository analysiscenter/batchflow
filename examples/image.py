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

    def _add_1(self, x):
        return x + 1

    @action
    def my_resize(self):
        #return self.apply_transform('images', 'images', scipy.ndimage.zoom, 2, order=3)
        fn = lambda data, factor: scipy.ndimage.zoom(data, factor, order=3)
        return self.apply_transform(None, 'images', np.ones, (S,S))

    @action
    def print(self):
        print("shape:", self.images.shape)
        print(self.images[0])


if __name__ == "__main__":
    # number of items in the dataset
    K = 6
    S = 12

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

    #res = ds_data.p.load(data).convert_to_PIL('images').resize((384, 384))
    #res = ds_data.p.load(data).resize((384, 384), method='PIL')
    res = ds_data.p.load(data).my_resize().print()

    print("Start...")
    t = time()
    res.run(K//2, n_epochs=1)
    print("End", time() - t)
