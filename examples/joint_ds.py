# pylint: skip-file
import os
import sys
import shutil
import numpy as np
import pandas as pd

sys.path.append('..')
from dataset import * # pylint: disable=wildcard-import


# Example of custome Batch class which defines some actions
class MyDataFrameBatch(DataFrameBatch):
    @action
    def print(self, ds=None, text=None):
        if text is not None:
            print(text)
        print(self.data)
        if ds is not None:
           print('Joined data')
           print(ds[0].data)
        return self

    @action
    def action1(self):
        print("action 1")
        return self

    @action
    def action2(self):
        print("action 2")
        return self

    @action
    def action3(self):
        print("action 3")
        return self

    @action
    def add(self, inc):
        self.data += inc
        return self


# remove directory with subdirectories
DIR_PATH = './data/dirs'
shutil.rmtree(DIR_PATH, ignore_errors=True)
# create new dirs
for i in range(3):
    for j in range(5):
        os.makedirs(os.path.join(DIR_PATH, 'dir' + str(i), str(i*5 + j)))


# Create index from ./data/dirs
dindex = FilesIndex(os.path.join(DIR_PATH, 'dir*/*'), dirs=True, sort=True)
# print list of subdirectories
print("Dir Index:")
print(dindex.index)

align = False
if align:
    oindex = DatasetIndex(np.arange(len(dindex))+100)
else:
    oindex = FilesIndex(os.path.join(DIR_PATH, 'dir*/*'), dirs=True, sort=False)
print("\nOrder Index:")
print(oindex.index)

ds1 = Dataset(dindex, MyDataFrameBatch)
ds2 = Dataset(oindex, MyDataFrameBatch)
jds = JointDataset((ds1,ds2), align='order' if align else 'same')

K = 5

print()
for b1, b2 in jds.gen_batch(K, one_pass=True):
	print(b1.index)
	print(b2.index)


print("\n\nSplit")
jds.cv_split([0.5, 0.35])
for dsi in [jds.train, jds.test, jds.validation]:
    if dsi is not None:
        print("Joint index:", dsi.index.index)
        b1, b2 = jds.create_batch(dsi.index.index)
        print("DS1:", b1.index)
        print("DS2:", b2.index)
        print()
