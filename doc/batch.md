# Batch class

Batch class holds the data and contains processing functions.
Normally, you never create batch instances, as they are created in the `Dataset` or `Pipeline` batch generators.


## Index
`Batch` class stores the index of all data items which belongs to the batch. You can access the index through `self.index`. The sequence of indices is also available as `self.indices`.


## Data
The base `Batch` class has a private property `_data` which you can use to store your data in. So you can access the data within batch class methods through `self._data`. But you may also create new properties for your specific data. For example, an image batch class for segmentation purposes may have properties `_images` and `_masks` which point to corresponding image arrays.

There is also a public property `data` defined as:
```python
@property
def data(self):
    return self._data
```
This approach lets to conceal an internal data structure and allow for a more convenient and (perhaps) more stable public interface for data access.

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
Take into account that an `action` method should return a batch instance of the very same class or some other class.
If an `action` changes the instance's data directly it may simply return `self`.

## Running methods in parallel
As batch can be large it might make sense to parallel the computations.
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
