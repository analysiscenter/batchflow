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
    @inbatch_parallel(init='indices', post='_assemble_batch')
    def load(self, ix, src, fmt=None):
        if fmt == 'PIL':
            return PIL.Image.fromarray(src[ix])
        else:
            return src[ix]

    @action
    @inbatch_parallel(init='get_image', post='_assemble_batch')
    def convert_to_PIL(self, image):
            return PIL.Image.fromarray(image.astype('unit8'))

    def _assemble_batch(self, all_res, *args, **kwargs):
        if any_action_failed(all_res):
            raise ValueError("Could not assemble the batch", self.get_errors(all_res))
        if isinstance(all_res[0], PIL.Image.Image):
            self.images = all_res
        else:
            self.images = np.transpose(np.dstack(all_res), (2, 0, 1))
        return self

    @action
    @inbatch_parallel(init='get_image', post='_assemble_batch')
    def resize(self, image, shape):
        if isinstance(image, PIL.Image.Image):
            return image.resize(shape, PIL.Image.ANTIALIAS)
        else:
            factor = 1. * np.asarray(shape) / np.asarray(image.shape)
            return scipy.ndimage.zoom(image, factor, order=3)

    @action
    @inbatch_parallel(init='get_image', post='_assemble_batch')
    def presize(self, image, shape):
        return PIL.Image.fromarray(image).resize(shape, PIL.Image.ANTIALIAS)


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

    res = ds_data.p.load(data).convert_to_PIL().resize((384, 384))

    print("Start...")
    t = time()
    res.run(K//10, n_epochs=1)
    print("End", time() - t)
