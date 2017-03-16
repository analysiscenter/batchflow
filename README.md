# Dataset

# Basic usage

```python
import numpy as np
import pandas as pd
import dataset as ds


# Batch class contains action methods to process data in one batch
class PatientScans(ds.Batch):
  ...
  @action
  def load(path):
  ...
  @action
  def resize(h, w, d):
  ...
  @action
  def segment_lungs():
  ...


CT_SCAN_PATH = '/path/to/CT_scans'
# each CT_SCAN_PATH's subdirectory contains one patient scans and its name is the patient id
patient_index = ds.FilesIndex(CT_SCAN_PATH, dirs=True, sort=False)
# The scan data is too huge to load into memory at once
# So we define a dataset which has an index and a specific action class which process a small subset of data (batch)
scans_dataset = ds.Dataset(index=patient_index, batch_class=PatientScans)

# Preprocessing allows to define the process workflow
# All the actions are lazy so that they are not executed unless their results are needed
scans_preprocessing = (ds.Preprocessing(scans_dataset).
                        load(path=CT_SCAN_PATH).  # you can build index from one dir and then load data from another dir
                        resize(256, 256, 128).    # resize the image
                        segment_lungs().          # remove from the image everything except the lungs
                        dump('/path/to/processed/scans')   # save preprocessed scans to disk
# load, resize, segment_lungs, and dump are the methods of PatientScans
# Preprocessing knows nothing about your data, where it is stored and how to process it.
# It's just a convenient wrapper.
# By the way, nothing has been executed yet. And there were no batches either.
# Everything is lazy!

# And now let's run it!
scans_preprocessing.run(batch_size=64, shuffle=False, one_pass=True)
# Inside run the dataset is divided into batches and all the actions are executed for each batch


# Moving further, you need to make a combined dataset which contains scans and labels
# So we create a dataset with labels as pandas.DataFrame
labels_dataset = ds.Dataset(index=patient_index, batch_class=DataFrameBatch)
# Define how you need to process labels (here you just load them from a file)
labels_processing = Preprocessing(labels_dataset).load('/path/to/labels.csv', fmt='csv')

# To train the model you might need some data augmentation
# Let's define it as a lazy process
scan_augm = (Preprocessing(scans_dataset).
               load('path/to/processed/scans').
               random_crop(64, 64, 64).
               random_rotate(-pi/4, pi/4))

# Define the combined dataset
full_data = ds.FullDataset(data=scan_augm, target=labels_processing)
# And again, nothing has been executed or loaded yet.

# Start the training
NUM_ITERS = 10000
for i in range(NUM_ITERS):
    # get random batches which contains some augmented scans and corresponding labels
    # all the actions are executed now
    bscan, blabel = full_data.next_batch(batch_size=32, shuffle=True)
    # put it into neural network
    sess.run(optimizer, feed_dict={x: bscan, y: blabel['labels'].values}))

```

For more advanced cases see [the documentation](doc/INDEX.md)
