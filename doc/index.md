# Index

A dataset may be so large that it does not fit into memory and thus you cannot process it at once. That is why each data item in the `Dataset` should have an unique id. It does not have to be meaningful (like a card number or a transaction id), sometimes it may be just a hash or an ordered number. However, each index item should address exactly one data item (which in turn can have a complex structure, like a list, an array, a dataframe, or even a graph).

## DatasetIndex

`DatasetIndex` is a base index class which stores a sequence of unique ids for your data items. In the simplest case it might be just an ordered sequence of numbers (0, 1, 2, 3,..., e.g. `numpy.arange(len(dataset))`).
```python
dataset_index = DatasetIndex(np.arange(my_array.shape[0]))
```

In other cases it can be a list of domain-specific identificators (e.g. client ids, product codes, serial numbers, timestamps, etc).
```python
dataset_index = DatasetIndex(dataframe['client_id'])
```

You will rarely need to work with an index directly, but if you want to do something specific you may use its public API.

### Public API

#### get_pos(item_id)
Returns the position of the `item_id` in the index sequence.
```python
#   positions            0           1          2          3
index = DatasetIndex(['item_01', 'item_02', 'item_03', 'item_04'])
pos = index.get_pos('item_03')
# pos will be equal 2
```

#### cv_split(shares)
Split index into train, test and validation subsets. Shuffles index if necessary.
Subsets are also `DatasetIndex` objects and are available as attributes `.train`, `.test` and `.validation` respectively.

Split into train / test in 80/20 ratio (default)
```python
index.cv_split()
```
Split into train / test / validation in 60/30/10 ratio
```python
index.cv_split([0.6, 0.3])
```
Split into train / test / validation in 50/30/20 ratio
```python
index.cv_split([0.5, 0.3, 0.2])
```

#### next_batch(batch_size, shuffle, n_epochs, drop_last)
Returns a batch from the index.

Args:
`batch_size` - number of items in each batch.

`shuffle` - whether to randomize items order before splitting into batches. Can be  
- `bool`: `True` / `False`
- a `RandomState` object which has an inplace shuffle method (see [numpy.random.RandomState](https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.RandomState.html)):
- `int` - a random seed number which will be used internally to create a `numpy.random.RandomState` object
- `sample function` - any callable which gets an order and returns a shuffled order.

Default - `False`.

`n_epochs` - number of iterations around the whole index. If None, then you will have an infinite sequence of batches. Default value - 1.

`drop_last` - whether to skip the last batch if it has fewer items (for instance, if an index contains 10 items and the batch size is 3, then there will 3 batches of 3 items and the last batch with just 1 item).

Returns:
an instance of DatasetIndex holding a subset of the original index

Usage:
```python
for i in range(MAX_ITERS):
    index_batch = index.next_batch(BATCH_SIZE, n_epochs=None)
```

#### gen_batch(batch_size, shuffle, n_epochs, drop_last)
Returns a batch generator.

Usage:
```python
for index_batch in index.gen_batch(BATCH_SIZE, shuffle=False, n_epochs=1):
    # do something
```

## FilesIndex
When data comes from a file system, it might be convenient to use `FilesIndex`.
```python
files_index = FilesIndex("/path/to/some/files/*.csv")
```
Thus `files_index` will contain the list of filenames that match a given mask.
The details of mask specification may be found in the [glob](https://docs.python.org/3/library/glob.html) documentation.

### No extension
When filenames contain extensions which are not a part of the id, then they may be stripped with an option `no_ext`:
```python
dataset_index = FilesIndex("/path/to/some/files/*.csv", no_ext=True)
```

### Sorting
Since order may be random, you may want to sort your index them:
```python
dataset_index = FilesIndex("/path/to/some/files/*.csv", sort=True)
```
However, this rarely makes any sense.

### Directories
Sometimes you need directories, not files. For instance, a CT images dataset includes one subdirectory per each patient, it is named with patient id and contains many images of that patient. So the index should be built from these subdirectories, and not separate images.
```python
dirs_index = FilesIndex("/path/to/archive/2016-*/scans/*", dirs=True)
```
Here `dirs_index` will contain an ordered list of all subdirectories names.

### Numerous sources
If files you are interested in are located in different places you may still build an index:
```python
dataset_index = FilesIndex(["/current/year/data/*", "/path/to/archive/2016/*", "/previous/years/*"])
```

### Public API
See above [DatasetIndex API](#public-api).

#### get_fullpath(item_id)
Returns the full path for the `item_id`.
```python
index = FilesIndex(['/some/path/*.csv', '/other/path/data/*'])
fullpath = index.get_fullpath('item_03')
```
`fullpath` will contain the fully qualified name like `/some/path/item_03.csv`.
