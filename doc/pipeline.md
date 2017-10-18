# Pipeline

## Content
1. [Introduction](#introduction)
1. [Algebra of pipelines](#algebra-of-pipelines)
1. [Creating pipelines](#creating-pipelines)
1. [Running pipelines](#running-pipelines)
1. [Pipeline variables](#pipeline-variables)
1. [Join and merge](#join-and-merge)
1. [Models](#models)
1. [Public API](#public-api)


## Introduction
Quite often you can't just use the data itself. You need to preprocess it beforehand. And not too rarely you end up with several processing workflows which you have to use simultaneously. That is the situation when pipelines might come in handy.

```python
class ClientTransactions(Batch):
    ...
    @action
    def some_action(self):
        ...
        return self

    @action
    def other_action(self, param):
        ...
        return self

    @action
    def yet_other_action(self):
        ...
        return self

```
To begin with, you create a dataset (client_index is an instance of [DatasetIndex](index.md)):
```python
ct_ds = Dataset(client_index, batch_class=ClientTranasactions)
```

And then you can define a workflow pipeline:
```python
trans_pipeline = (ct_ds.pipeline()
                    .some_action()
                    .other_action(param=2)
                    .yet_other_action())
```
And nothing happens! Because all the actions are lazy.
Let's run them.
```python
trans_pipeline.run(BATCH_SIZE, shuffle=False, n_epochs=1)
```
Now the dataset is split into batches and then all the actions are executed for each batch independently.

In the very same way you can define an augmentation workflow
```python
augm_wf = (image_dataset.pipeline()
            .load('/some/path')
            .random_rotate(angle=(-30, 30))
            .random_resize(factor=(0.8, 1.2))
            .random_crop(factor=(0.5, 0.8))
            .resize(shape=(256, 256))
)
```
And again, no action is executed until its result is needed.
```python
NUM_ITERS = 1000
for i in range(NUM_ITERS):
    image_batch = augm_wf.next_batch(BATCH_SIZE, shuffle=True, n_epochs=None)
    # only now the actions are fired and data is changed with the workflow defined earlier
```

## Algebra of pipelines
There are two ways to define a pipeline:
- a chain of actions
- a pipeline algebra

An action chain is a concise and convenient way to write pipelines. But sometimes it's not enough, for instance, when you want to manipulate with many pipelines adding them or multiplying as if they were numbers or matrices. And that's what we call `a pipeline algebra`.

There are 5 operations available: `+`, `*`, `@`, `<<` , `>>`.

### + (concat)
Add two pipelines by concatenating them, so the actions from the first pipeline will be executed before actions from the second one.
`p.resize(shape=(256, 256)) + p.rotate(angle=45)`

### * (repeat)
Repeat the pipeline several times.
`p.random_rotate(angle=(-30, 30)) * 3`

### @ (sometimes)
Execute the pipeline with the given probability.
`p.random_rotate(angle=(-30, 30)) @ 0.5`

### `>>` and `<<`
Link a pipeline to a dataset.
`dataset >> pipeline` or `pipeline << dataset`

The complete example:
```python
from dataset import Pipeline

with Pipeline() as p:
    preprocessing_pipeline = p.load('/some/path') +
                             p.resize(shape=(256, 256)) +
                             p.random_rotate(angle=(-30, 30)) @ .8 +
                             p.random_transform() * 3 +
                             p.random_crop(shape=(128, 128))
images_prepocessing = preprocessing_pipeline << images_dataset
```

## Creating pipelines
Pipelines can be created from scratch or from a dataset.

### A template pipeline
```python
from dataset import Pipeline

my_pipeline = Pipeline()
                .some_action()
                .another_action()
```
Or through a context manager with pipeline algebra:
```python
from dataset import Pipeline

with Pipeline() as p:
    my_pipeline = p.some_action() +
                  p.another_action()
```
However, you cannot execute this pipeline as it doesn't linked to any dataset.
On the other hand, such pipelines might be applied to different datasets:
```python
cifar10_pipeline = template_preprocessing_pipeline << cifar10_dataset
mnist_pipeline = template_preprocessing_pipeline << mnist_dataset
```

### A dataset pipeline
```python
my_pipeline = my_dataset.pipeline()
                .some_action()
                .another_action()
```
Or a shorter version:
```python
my_pipeline = my_dataset.p
                .some_action()
                .another_action()
```
Every call to `dataset.pipeline()` or `dataset.p` creates a new pipeline.

## Running pipelines
There are 4 ways to execute a pipeline.

### Batch generator
```python
for batch in my_pipeline.gen_batch(BATCH_SIZE, shuffle=True, n_epochs=2, drop_last=True):
    # do whatever you want
```
`batch` will be the batch returned from the very last action of the pipeline.

The important note:  
`BATCH_SIZE` is a size of the batch taken from the dataset. Actions might change the size of the batch and thus
the batch you will get from the pipeline might have a different size.

### next_batch function
```python
for i in range(MAX_ITER):
    batch = my_pipeline.next_batch(BATCH_SIZE, shuffle=True, n_epochs=2, drop_last=True)
    # do whatever you want
```

### Run
```python
my_pipeline = (dataset.p
                  .some_action()
                  .other_action()
                  .yet_another_action()
                  .run(BATCH_SIZE, n_epochs=2, drop_last=True))
```

### Lazy run
```python
my_pipeline = (dataset.p
                  .some_action()
                  .other_action()
                  .yet_another_action()
                  .run(BATCH_SIZE, n_epochs=None, drop_last=True, lazy=True)
)

for i in range(MAX_ITER):
    batch = my_pipeline.next_batch()
    # do whatever you want
```
You can add `run` with `lazy=True` as the last action in the pipeline and then call `run()` or `next_batch()` without arguments at all.


## Pipeline variables
Sometimes batches can be processed in a "do and forget" manner: when you take a batch, make some data transformations and then switch to another batch.
However, not infrequently you might need to remember some parameters or intermediate results (e.g. a value of loss function or accuracy on every batch
to draw a graph later). This is why you might need pipeline variables.

### Initializing a variable
```python
my_pipeline = my_dataset.p
                 .init_variable("my_variable", 100)
                 .init_variable("some_counter", init_on_each_run=0)
                 .init_variable("var with init function", init=my_init_function)
                 .init_variable("loss_history", init_on_each_run=list)
                 .first_action()
                 .second_action()
                 ...
```
To initialize a variable just add to a pipeline `init_variable(...)` with a variable name and a default value.
Variables might be initialized once in a lifetime (e.g. some global state or a configuration parameter) or before each run
(like counters or history stores).

Sometimes it is more convenient to initialize variables indirectly through a function. For instance, `loss_history` cannot be initialized with `[]`
as it would make a global variable which won't be cleared on every run. What you actually need is a call to `list()` on each run.

Init functions are also a good place for some complex logic or randomization.

### Updating a variable
Each batch instance have a pointer to the pipeline it was created in (or `None` if the batch was created manually).

So getting an access to a variable is easy:
```python
class MyBatch(Batch):
    ...
    @action
    def some_action(self):
        var_value = self.pipeline.get_variable("variable_name")
        ...
```
If a variable does not exist, it might be created and initialized, if `create` parameter is set to `True`.
For a flexible initialization `default`, `init` and `init_on_each_run` might also be passed to `get_variable()`.

If `create` is `False` (which is by default), then `get_variable` will raise a `KeyError` if a variable does not exist.

To change a variable value just call `set_variable` within an action:
```python
class MyBatch(Batch):
    ...
    @action
    def some_action(self):
        ...
        self.pipeline.set_variable("variable_name", new_value)
        ...
```

Or add `update_variable` to the pipeline:
```python
my_pipeline
    ...
    .update_variable("current_batch_labels", fn=MyBatch.get_labels)
    .update_variable("all_labels", var='current_batch_labels', mode='append')
```
If `fn` is specified, it would be called with the current batch as a parameter.

Mode 'append' allows to gather values from all the batches. Append mode works with both `var` and `fn` parameters.


### Deleting a variable
Just call `pipeline.delete_variable("variable_name")` or `pipeline.del_variable("variable_name")`.

### Deleting all variables
As simple as `pipeline.delete_all_variables()`

### Variables as locks
If you use multi-threading [prefetching](prefetch.md) or [in-batch parallelism](parallel.md),
than you might require synchronization when accessing some shared resource.
And pipeline variables might be a handy place to store locks.
```python
class MyBatch(Batch):
    ...
    @action
    def some_action(self):
        ...
        with self.pipeline.get_variable("my lock"):
            # only one some_action will be executing at this point
    ...

my_pipeline = my_dataset.p
                .init_variable("my lock", init=threading.Lock)
                .some_action()
                ...

```


## Join and merge

### Joining pipelines
If you have a pipeline `images` and a pipeline `labels`, you might join them for a more convenient processing:
```python
images_with_labels = (images.p
                            .load(...)
                            .resize(shape=(256, 256))
                            .random_rotate(angle=(-pi/4, pi/4))
                            .join(labels)
                            .some_action())
```
When this pipeline is run, the following will happen for each batch of `images`:
- the actions `load`, `resize` and `random_rotate` will be executed
- a batch of `labels` with the same index will be created
- the `labels` batch will be passed into `some_action` as a first argument (after `self`, of course).

So, images batch class should look as follows:
```python
class ImagesBatch(Batch):
    def load(self, src, fmt):
        ...

    def resize(self, shape):
        ...

    def random_rotate(self, angle):
        ...

    def some_actions(self, labels_batch):
        ...
```

You can join several sources:
```python
full_images = (images.p
                     .load(...)
                     .resize(shape=(256, 256))
                     .random_rotate(angle=(-30, 30))
                     .join(labels, masks)
                     .some_action())
```
Thus, the tuple of batches from `labels` and `masks` will be passed into `some_action` as the first arguments (as always, after `self`).

Mostly, `join` is used as follows:
```python
full_images = (images.p
                     .load(...)
                     .resize(shape=(256, 256))
                     .join(labels, masks)
                     .load(components=['labels', 'masks']))
```
See [batch.load](batch.md#load) for more details.


### Merging pipelines
You can also merge data from two pipelines (this is not the same as [concatenating pipelines](#algebra-of-pipelines)).
```python
images_with_augmentation = (images_dataset.p
                               .load(...)
                               .resize(shape=(256, 256))
                               .random_rotate(angle=(-30, 30))
                               .random_crop(shape=(128, 128))
                               .run(batch_size=16, epochs=None, shuffle=True, drop_last=True, lazy=True)

all_images = (images_dataset.p
                   .load(...)
                   .resize(shape=(128, 128))
                   .merge(images_with_augmentation)
                   .run(batch_size=16, epochs=3, shuffle=True, drop_last=True)
```
What will happen here is
- `images_with_augmentation` will generate batches of size 16
- `all_images` before merge will generate batches of size 16
- `merge` will combine both batches in some way.

Pipeline's `merge` calls `batch_class.merge([batche_from_pipe1, batch_from_pipe2])`.

The default `Batch.merge` just concatenate data from both batches, thus making a batch of double size.

Take into account that the default `merge` also changes index to `numpy.arange(new_size)`.


## Rebatch
When actions change the batch size (for instance, dropping some bad or skipping incomplete data),
you might end up in a situation when you don't know the batch size and, what is sometimes much worse,
batch size differs. To solve this problem, just call `rebatch`:
```python
images_pipeline = (images_dataset.p
                       .load(...)
                       .random_rotate(angle=(-30, 30))
                       .skip_black_images()
                       .skip_too_noisy_images()
                       .rebatch(32)
```
Under the hood `rebatch` calls `merge`, so you must ensure that `merge` works properly for your specific data and write your own `merge` if needed.


## Models
See [Working with models](models.md)


## Public API

### `gen_batch(batch_size, shuffle=True, n_epochs=1, drop_last=False, prefetch=0)`
Returns a batch generator.

Usage:
```python
for batch in my_pipeline.gen_batch(BATCH_SIZE, shuffle=True, n_epochs=1):
    # do something
```

### `next_batch(batch_size, shuffle=True, n_epochs=1, drop_last=False, prefetch=0)`
Gets a batch from the dataset, executes all the actions defined in the pipeline and then returns the result of the last action.

Args:
`batch_size` - number of items in each batch.

`shuffle` - whether to randomize items order before splitting into batches. Can be  
- `bool`: `True` / `False`
- a `RandomState` object which has an inplace shuffle method (see [numpy.random.RandomState](https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.RandomState.html)):
- `int` - a random seed number which will be used internally to create a `numpy.random.RandomState` object
- `sample function` - any callable which gets an order and returns a shuffled order.

Default - `False`.

`n_epochs` - number of iterations around the whole index. If `None`, then you will get an infinite sequence of batches. Default value - 1.

`drop_last` - whether to skip the last batch if it has fewer items (for instance, if an index contains 10 items and the batch size is 3, then there will 3 batches of 3 items and the last batch with just 1 item).

`prefetch` - the number of batches processed in advance (see [details](prefetch.md))

Returns:
an instance of the batch class returned from the last action in the pipeline

Usage:
```python
for i in range(MAX_ITERS):
    batch = my_pipeline.next_batch(BATCH_SIZE, n_epochs=None)
```

### `run(batch_size, shuffle=True, n_epochs=1, drop_last=False, prefetch=0, lazy=False)`
Runs a pipeline.
However, when `lazy=True`, just remembers parameters for a later `run()` or `next_batch()` without arguments.

If `lazy=False`, `run` is:
```python
for _ in my_pipeline.gen_batch(...):
    pass
```

### `join(another_source, one_more_source, ...)`
Joins corresponding batches from several sources (datasets or pipelines).

### `merge(another_source, one_more_source, ...)`
Merges batches from several sources (datasets or pipelines).

### `rebatch(batch_size)`
Splits and merges batches coming from the previous actions to form a batch of a given size.

### `init_model(mode, model_class=None, model_name=None, config=None)`
Initialize [a model](models.md)

### `import_model(model_name, from_pipeline)`
Import a static or dynamic model from another pipeline.

### `train_model(model_name, save_to=None, append_to=None, *args, **kwargs)`
Train a model during pipeline execution.

### `predict_model(model_name, save_to=None, append_to=None, *args, **kwargs)`
Predict using a model during pipeline execution.

### `init_variable(name, default=None, init=None, init_on_each_run=None)`
Create a variable with the default value or init function.

### `get_variable(name, default=None, init=None, init_on_each_run=None, create=False)`
Return a value of the variable with a given name (might create a variable if it does not exist)

### `set_variable(name, value)`
Immediately set a new value for a variable.

Same as `assign_variable()`

### `assign_variable(name, value)`
Immediately set a new value for a variable.

Same as `set_variable()`

### `update_variable(name, value=None, fn=None, var=None)`
Update a variable value during pipeline execution.

### `del_variable(name)`
Deletes a variable with a given name.

### `delete_variable(name)`
Deletes a variable with a given name.
