# Batch class

Batch class holds the data and contains processing functions.
Normally, you never create batch instances, as they are created in the `Dataset` or `Pipeline` batch generators.


## Index
`Batch` class stores the [index](index.md) of all data items which belong to the batch. You can access the index through `self.index`. The sequence of indices is also available as `self.indices`.


## Data
The base `Batch` class has a private property `_data` which you can use to store your data in. So you can access the data within batch class methods through `self._data`. But you may also create new properties for your specific data. For example, an image batch class for segmentation purposes may have properties `_images` and `_masks` which point to corresponding image arrays.

There is also a public property `data` defined as:
```python
@property
def data(self):
    return self._data
```
This approach allows to conceal an internal data structure and provides for a more convenient and (perhaps) more stable public interface to access the data.

An earlier mentioned batch with images may redefine `data` as:
```python
@property
def data(self):
    return self._images, self._masks
```

Nevertheless, this is just a convention and you are not obliged to follow it.


## Action methods
`Action` methods form a public API of the batch class which is available in the [pipeline](pipeline.md). If you operate directly with the batch class instances you don't need `action` methods. However, pipelines provide the most convenient interface to process the whole dataset and to separate data processing steps and model training / validation cycles.

In order to convert a batch class method to an action you add `@action` decorator:
```python
from dataset import Batch, action

class MyBatch(Batch):
    ...
    @action
    def some_action(self):
        # process your data
        return self
```
Take into account that an `action` method should return an instance of some `Batch`-class: the very same or some other class.
If an `action` changes the instance's data directly it may simply return `self`.

## Running methods in parallel
As a batch can be quite large it might make sense to parallel the computations. And it is pretty easy to do:
```python
from dataset import Batch, inbatch_parallel, action

class MyBatch(Batch):
    ...
    @action
    @inbatch_parallel(init='_init_fn', post='_post_fn', target='threads')
    def some_action(self, item, arg1, arg2):
        # process just one item
        return some_value
```
For further details how to make parallel actions see [parallel.md](parallel.md).


## Writing your own Batch

### Constructor should include `*args` and `*kwargs`
```python
class MyBatch(Batch):
    ...
    def __init__(self, index, your_param1, your_param2, *args, **kwargs):
        super().__init__()
        # process your data
```

### Don't load data in the constructor
The constructor should just intialize properties.
`Action`-method `load` is the best place for reading data from files or other sources.

So DON'T do this:
```python
class MyBatch(Batch):
    ...
    def __init__(self, index, your_param1, your_param2, *args, **kwargs):
        super().__init__()
        ...
        self._data = read(file)
```

Instead DO that:
```python
class MyBatch(Batch):
    ...
    def __init__(self, index, your_param1, your_param2, *args, **kwargs):
        super().__init__()
        ...

    @action
    def load(self, source, format):
        # load data from source
        ...
        self._data = read(file)
        return self
```

### (optional) Store your data in `_data` property
It is just a convenient convention which makes your life more consistent.

### (optional) Define `__getitem__` method
If you want to address batch items easily as well as iterate over your batch, you need `__getitem__` method. The default `__getitem__` from a base `Batch` looks like this:
```python
class MyBatch(Batch):
    ...
    def __getitem__(self, item):
        return self.data[item]
```
Thus you will be able to address batch items as `self[index_id]` internally (in the batch class methods) and as `batch[index_id]` externally.

###  Make all public methods `actions`
```python
class MyBatch(Batch):
    ...
    @action
    def change_data(self, item, arg1, arg2):
        # process your data
        return self
```
`Actions` should return an instance of some batch class.

### Parallel everyting you can
If you want a really fast data processing you can't do without `numba` or `cython`.
And don't forget about input/output operations.

### Make all I/O in `async` methods
This is extremely important if you read data from many files.
```python
class MyBatch(Batch):
    ...
    @action
    def load(self, format='raw'):
        if format == 'raw':
            self._data = self._load_raw()
        elif format == 'blosc':
            self._data = self._load_blosc()
        else:
            raise ValueError("Unknown format '%s'" % format)
        return self

    @inbatch_parallel(init='_init_io', post='_post_io', target='async')
    async def _load_raw(self, item):
        # load one data item from a raw format file
        return loaded_item

    def _init_io(self):
        return [[item_id, self.index.get_fullpath(item_id)] for item_id in self.indices]

    def _post_io(self, all_res, source):
        if any_action_failed(all_res):
            raise IOError("Could not load data from " + source)
        else:
            self._data = np.conatenate(all_res)
        return self
```
