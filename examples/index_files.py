# pylint: skip-file
import os
import sys
import shutil
import numpy as np
import pandas as pd

sys.path.append('..')
from dataset import * # pylint: disable=wrong-import-


# Create index from ./data
findex = FilesIndex('./data/*')
# print list of files
print("Index:")
print(findex.index)

print("\nSplit")
findex.cv_split([0.35, 0.35])
for dsi in [findex.train, findex.test, findex.validation]:
    if dsi is not None:
        print(dsi.index)

print("\nprint batches:")
for dsi in [findex.train, findex.test, findex.validation]:
    print("---")
    for b in dsi.gen_batch(2, one_pass=True):
        print(b)


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
print("Index:")
print(dindex.index)

print("\nSplit")
dindex.cv_split([0.35, 0.35])
for dsi in [dindex.train, dindex.test, dindex.validation]:
    if dsi is not None:
        print(dsi.index)
