# pylint: skip-file
import os
import sys
import numpy as np
from time import time

sys.path.append("../..")
from dataset import DatasetIndex, Dataset, Batch, action, inbatch_parallel, any_action_failed


# Example of custom Batch class which defines some actions
class MyBatch(Batch):
    @property
    def components(self):
        return "images", "labels"

    @action
    def print(self):
        print(self.items)
        return self

    @action
    @inbatch_parallel('indices', target='for')
    def other(self, ix):
        item = self[ix]
        pos = self.get_pos(None, 'images', ix)
        print("other:", ix, pos, type(item), item.images.ndim)


    @action
    @inbatch_parallel('items', target='for')
    def some(self, item):
        print("some:", type(item), item.images.ndim)


if __name__ == "__main__":
    # number of items in the dataset
    K = 4
    S = 12

    # Fill-in dataset with sample data
    def gen_data():
        ix = np.arange(K)
        images = np.random.randint(0, 255, size=K*S*S).reshape(-1, S, S).astype('uint8')
        labels = np.random.randint(0, 3, size=K).astype('uint8')
        data = images, labels

        ds = Dataset(index=ix, batch_class=MyBatch, preloaded=data)
        return ds, data


    # Create datasets
    print("Generating...")
    ds_data, data = gen_data()

    res = ds_data.p.print().other().some()

    print("Start...")
    t = time()
    res.run(2, n_epochs=1, prefetch=0, target='threads')
    print("End", time() - t)
