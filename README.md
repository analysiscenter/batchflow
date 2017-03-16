# Dataset

# Basic usage

A dataset consists of an index (1-d sequence with unique keys per each data item)
and a batch class which process the data.

There are some ready-to-use indices (e.g. `FilesIndex`) and batch classes (e.g. `ArrayBatch` and `DataFrameBatch`),
but you are likely to need your own classes with specific action methods.

So let's define a class to work with your unique data (for instance, CT scans)
```python
import numpy as np
import pandas as pd
import dataset as ds

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
```

The scans are too huge to load into memory at once.
That is why we need the `Dataset` class which holds an index of all data items (scans) and a specific action class to process a small subset of scans (batch)
```python
CT_SCAN_PATH = '/path/to/CT_scans'
# each CT_SCAN_PATH's subdirectory contains one patient scans and its name is the patient id
patient_index = ds.FilesIndex(CT_SCAN_PATH, dirs=True, sort=False)
scans_dataset = ds.Dataset(index=patient_index, batch_class=PatientScans)
```

In real life we rarely have a clean and ready-to-use dataset. 
Usually we have to make some preprocessing - a workflow of actions.
That is where the batch class comes into handy.
```python
scans_pp = (ds.Preprocessing(scans_dataset).
                load(path=CT_SCAN_PATH).  # load scans for one batch
                resize(256, 256, 128).    # resize the image
                segment_lungs().          # remove from the image everything except the lungs
                dump('/path/to/processed/scans'))   # save preprocessed scans to disk
```
All the actions are lazy so they are not executed unless their results are needed.

Take into account that `load`, `resize`, `segment_lungs`, and `dump` are `PatientScans` methods marked with `@action` decorator.

`Preprocessing` knows nothing about your data, where it is stored and how to process it.
It's just a convenient wrapper.

By the way, as nothing has been executed yet, there were no batches either.
Everything is lazy!

So let's run it!
```python
scans_pp.run(batch_size=64, shuffle=False, one_pass=True)
```
Inside `run` the dataset is divided into batches and all the actions are executed for each batch.

Moving further, you might want to make a combined dataset which contains scans and labels.
And labels might come as a `pandas.DataFrame` loaded from a `csv`-file.
```python
# labels have the very same index as scans
labels_dataset = ds.Dataset(index=patient_index, batch_class=DataFrameBatch)
# Define how you need to process labels (here you just load them from a file)
labels_processing = Preprocessing(labels_dataset).load('/path/to/labels.csv', fmt='csv')
```

You often train a model with augmented data. Let's configure it as a lazy process too.
```python
scan_augm = (Preprocessing(scans_dataset).
               load('/path/to/processed/scans').
               random_crop(64, 64, 64).
               random_rotate(-pi/4, pi/4))
```

Now define the combined dataset
```python
full_data = ds.FullDataset(data=scan_augm, target=labels_processing)
```
And again, nothing has been executed or loaded yet.
You don't have to waste CPU, GPU or IO cycles unless you need the processed data.

Before training you can split the dataset into train / test / validation subsets.
```python
full_data.cv_split([0.7, 0.2])
```

And start the training 
```python
NUM_ITERS = 10000
for i in range(NUM_ITERS):
    # get random batches which contains some augmented scans and corresponding labels
    # all the actions are executed now
    bscans, blabel = full_data.train.next_batch(batch_size=32, shuffle=True)
    # put it into neural network
    sess.run(optimizer, feed_dict={x: bscans, y: blabel['labels'].values}))

```

For more advanced cases and detailed API see [the documentation](doc/INDEX.md)
