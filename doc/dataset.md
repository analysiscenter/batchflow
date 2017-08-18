# Dataset

The `Dataset` holds an index of all data items (e.g. customers, transactions, etc) and a specific action class to process a small subset of data (batch)
```python
import dataset as ds

client_ix = ds.DatasetIndex(client_data.index)
client_ds = ds.Dataset(client_ix, batch_class=ds.DataFrameBatch)
```

And now you can conveniently iterate over the dataset
```python
BATCH_SIZE = 200
for client_batch in client_ds.gen_batch(BATCH_SIZE, shuffle=False, n_epochs=1):
    # client_batch is an instance of DataFrameBatch which holds an index of the subset of the original dataset
    # so you can do anything you want with that batch
    # for instance, load some data, as the batch is empty when initialized
    batch_with_data = client_batch.load(client_data)
```
You can define a new [batch class](batch.md) with action methods to process your specific data.

For machine learning models you might also need to generate random batches with `gen_batch` or `next_batch`:
```python
NUM_ITERS = 1000
for i in range(NUM_ITERS):
    client_batch = client_ds.next_batch(BATCH_SIZE, shuffle=True, n_epochs=None)
    # ...
```

## Public API

### `__init__(index, batch_class, preloaded=None)`
Creates a dataset with given `index` and `batch_class`.
```python
my_dataset = Dataset(some_index, batch_class=MyBatchClass)
```
If `preloaded` is specified, than each created batch will preload data from this argument.

### `next_batch(batch_size, shuffle=False, n_epochs=1, drop_last=False)`
Returns the next batch from the dataset

Args:
`batch_size` - number of items in the batch.

`shuffle` - whether to randomize items order before splitting into batches. Can be  
- `bool`: `False` - to make batches in the order of indices in the index, `True` - to make batches with random indices.
- a `RandomState` object which has an inplace shuffle method (see [numpy.random.RandomState](https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.RandomState.html)):
- `int` - a random seed number which will be used internally to create a `numpy.random.RandomState` object
- `sample function` - any callable which gets an order and returns a shuffled order.

Default - `False`.

`n_epochs` - number of iterations around the whole dataset. If `None`, then you will get an infinite sequence of batches. Default value - 1.

`drop_last` - whether to skip the last batch if it has fewer items (for instance, if a dataset contains 10 items and the batch size is 3, then there will 3 batches of 3 items and the 4th batch with just 1 item. The last batch will be skipped if `drop_last=True`).

Returns:
an instance of the batch class

Usage:
```python
for i in range(MAX_ITERS):
    batch = my_dataset.next_batch(BATCH_SIZE, n_epochs=None)
```

### `gen_batch(batch_size, shuffle=False, n_epochs=1, drop_last=False)`
Returns a batch generator.

Usage:
```python
for batch in my_dataset.gen_batch(BATCH_SIZE, shuffle=True, n_epochs=1):
    # do something
```
