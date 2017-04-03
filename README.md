# Dataset

## Basic usage

A dataset consists of an index (1-d sequence with unique keys per each data item)
and a batch class which process the data.

There are some ready-to-use indices (e.g. `FilesIndex`) and batch classes (e.g. `ArrayBatch` and `DataFrameBatch`),
but you are likely to need your own classes with specific action methods.


### Dataset

The `Dataset` holds an index of all data items (e.g. clients) and a specific action class to process a small subset of data (batch)
```python
import pandas as pd
import dataset as ds


client_data = pd.read_csv("/path/to/some/data.csv")
# the simplest case: dataset index is just a dataframe index
client_ix = ds.DatasetIndex(client_data.index)
client_ds = ds.Dataset(client_ix, batch_class=ds.DataFrameBatch)
```

And now you can conveniently iterate over the dataset
```python
BATCH_SIZE = 200
for client_batch in client_ds.gen_batch(BATCH_SIZE, shuffle=False, one_pass=True)
    # client_batch is an instance of DataFrameBatch which holds an index of the subset of the original dataset
    # so you can do anything you want with that batch
    # for instance, load data from some source
    batch_df = client_batch.load(client_data)
```
You can define a new batch class which action methods process your specific data.

For machine learning models you might also need to generate random batches with `gen_batch` or `next_batch`:
```python
NUM_ITERS = 1000
for i in range(NUM_ITERS):
    client_batch = client_ds.next_batch(BATCH_SIZE, shuffle=True, one_pass=False)
    # ...
```


### DatasetIndex

`DatasetIndex` stores a sequence of unique ids for your data items. In the simplest case it might be just an ordered sequence of numbers (1, 2, 3,...).
In other cases it can be the list of domain-specific identificators (e.g. client ids, product codes, serial numbers, timestamps, etc).
When your data is stored in files it might be convenient to use `FilesIndex`
```python
files_index = FilesIndex("/path/to/some/files/*.csv", dirs=False, no_ext=True, order=False)
```
Thus `files_index` will contain the list of filenames without extensions.

Sometimes you might need to build an index from the list of subdirectories
```python
dirs_index = FilesIndex("/path/to/archive/2016-*/scans/*", dirs=True, order=True)
```
Here `dirs_index` will contain an ordered list of all subdirectories names.


### Processing workflow

Quite often you can't just use the data itself. You need to preprocess it beforehand. And not too rarely you end up with several processing workflows which you have to use simultaneously. That is the situation when Dataset might come in handy.

```python
class ClientTransactions(ds.DataFrameBatch):
  ...
  @action
  def show(self):
      print(self.data)
      return self

  @action
  def add(self, num=1):
      self.data = self.data + num
      return self

  @action
  def other_action(self):
      self.data = self.data + 10
      return self

```
If `client_index` contains the list of all clients ids, you can easily create a dataset
```python
ct_ds = Dataset(client_index, batch_class=ClientTranasactions)
```

And then you can define a workflow
```python
pp_wf = (ct_ds.workflow()
              .load(data)
              .print()
              .add()
              .print()
              .add(5)
              .print()
              .other_action()
              .print())
```
And nothing happens! Because all the actions are lazy.
Let's run them.
```python
pp_wf.run(BATCH_SIZE, shuffle=False, one_pass=True)
```
Now the dataset is split into batches and then all the actions are executed for each batch independently.

In the very same way you might define an augmentation workflow
```python
augm_wf = (ct_ds.workflow()
                .load(data)
                .add(1)
                .add(5)
                .print())
```
No action is executed until its result is needed.
```python
NUM_ITERS = 1000
for i in range(NUM_ITERS):
    client_batch = augm_wf.next_batch(BATCH_SIZE, shuffle=True, one_pass=False)
    # only now the actions are fired and data is changed with the workflow defined earlier
```
The original data stored in `ct_ds` is left unchanged. You can call `ct_ds.next_batch` in order to iterate over the source dataset without any augmentation.


For more advanced cases and detailed API see [the documentation](doc/index.md)
