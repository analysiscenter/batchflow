# Within batch parallelism

## Content
1. [Basic usage](#basic-usage)
1. [Parallelized methods](#parallelized-methods)
1. [Decorator arguments](#decorator-arguments)
1. [Additional decorator arguments](#additional-decorator-arguments)
1. [Init function](#init-function)
1. [Post function](#post-function)
1. [Targets](#targets)
1. [Arguments with default values](#arguments-with-default-values)
1. [Examples](#examples)


## Basic usage
In order to run a method in parallel you need to add `inbatch_parallel` decorator :

```python
from dataset import Batch, inbatch_parallel, action

class MyBatch(Batch):
    ...
    @action
    @inbatch_parallel(init='_init_default', post='_post_default', target='threads')
    def some_action(self, item, arg1, arg2):
        # process just one item
        return some_value
```

## Parallelized methods
You can parallelize actions as well as ordinary methods.

## Decorator arguments

### init='some_method'
Required.
The only required argument which contains a method name to be called to initialize the parallel execution.

### post='other_method'
Optional.
A method name which is called after all parallelized functions are finished.
### target='threads'
Optional.
Specifies a parallelization engine, should be one of `threads`, `nogil`, `async`, `mpc`.


## Additional decorator arguments
You can pass any other arguments to the decorator and they will be passed further to init and post functions.

```python
class MyBatch(Batch):
...
    @inbatch_parallel(init='_init_default', post='_post_default', target='threads', clear_data=False)
    def some_method(self, item):
        # process just one item
        return some_value

    def _init_default(self, clear_data):
        ...

    def _post_default(self, list_of_res, clear_data):
        ...
```

All these arguments should be named argments only. So you should not write like this:
```python
@inbatch_parallel('_init_default', '_post_default', 'threads', clear_data)
```
It might sometimes works though. But no guarantees.

The preferred way is:
```python
@inbatch_parallel(init='_init_default', post='_post_default', target='threads', clear_data=False)
```

Using this technique you can pass an action name to the `init` function:
```python
class MyBatch(Batch):
...
    @inbatch_parallel(init='_init_default', post='_post_default', target='threads', method='one_method')
    def one_method(self, item):
        # process just one item
        return some_value

    @inbatch_parallel(init='_init_default', post='_post_default', target='threads', method='some_other_method')
    def some_other_method(self, item):
        # process just one item
        return some_value
```
However, usually you might consider writing specific init / post functions for different actions.


## Init function
Init function defines how to parallelize the action. It returns a list of arguments for each invocation of the parallelized action.
So if you want to run 10 parallel copies of the method, `init` should return a list of 10 items. Usually you run the method once for each item in the batch. However you might also run one method per 10 or 100 or any other number of items if it is beneficial for your specific circumstances (memory, performance, etc.)

The simplest `init` just returns a sequence of indices:
```python
class MyBatch(Batch):
...
    def _init_default(self, *args, **kwargs):
        return self.indices

    @action
    @inbatch_parallel(init='_init_default')
    def some_action(self, item_id)
        # process an item and return a value for that item
        return proc_value
```
For a batch of 10 items `some_action` will be called 10 times as `some_action(index1)`, `some_action(index2)`, ..., `some_action(index10)`.

You may define as many arguments as you need:
```python
class MyBatch(Batch):
...
    def _init_default(self, *args, **kwargs):
        all = []
        for item in self.indices:
            ...
            item_args = [self._data, item, another_arg, one_more_arg]
            all.append(item_args)
        return all
```
Here the action will be fired as:  
`some_action(self._data, index1, another_arg, one_more_arg)`  
`some_action(self._data, index2, another_arg, one_more_arg)`  
`...`

`item_args` does not have to be strictly a list, but any sequence - tuple, numpy array, etc - that supports the unpacking operation ([`*seq`](https://docs.python.org/3/tutorial/controlflow.html#unpacking-argument-lists)):

**Attention!** It cannot be a tuple of 2 arguments (see below why).

You can also pass named arguments:
```python
class MyBatch(Batch):
...
    def _init_default(self, *args, **kwargs):
        all = []
        for item in self.indices:
            ...
            item_args = dict(data=self._data, item=item, arg1=another_arg, arg2=one_more_arg)
            all.append(item_args)
        return all
```
And the action will be fired as:  
`some_action(data=self._data, item=index1, arg1=another_arg, arg2=one_more_arg)`  
`some_action(data=self._data, item=index2, arg1=another_arg, arg2=one_more_arg)`  
`...`

And you can also combine positional and named arguments:
```python
class MyBatch(Batch):
...
    def _init_default(self, *args, **kwargs):
        all = []
        for item in self.indices:
            ...
            item_args = tuple(list(self._data, item), dict(arg1=another_arg, arg2=one_more_arg))
            all.append(item_args)
        return all
```
So the action will be fired as:  
`some_action(self._data, index1, arg1=another_arg, arg2=one_more_arg)`  
`some_action(self._data, index2, arg1=another_arg, arg2=one_more_arg)`  
`...`

Thus 2-items tuple is reserved to this situation (1st item is a list of positional arguments and 2nd is a dict of named arguments).

That is why you cannot pass a tuple of 2 arguments:
```python
    ...
    item_args = tuple(some_arg, some_other_arg)
    ...
```
Instead make it a list:
```python
    ...
    item_args = list(some_arg, some_other_arg)
    ...
```

### Init's additional arguments
Take into account that all arguments passed into actions are also passed into `init` function. So when you call:
```python
some_pipeline.some_parallel_action(10, 12, my_arg=12)
```
The `init` function will be called like that:
```python
init_function(10, 12, my_arg=12, arg_from_parallel_decorator=True)
```
This is convenient when you need to initialize some additional variables depending on the arguments. For instance, to create a numpy array of a certain shape filled with specific values or set up a random state or even pass additional arguments back to action methods.


## Post function
When all parallelized invocations are finished the `post` function is called.

The first argument it receives is the list of results from each parallelized function.
```python
class MyBatch(Batch):
    ...
    def _init_default(self, *args, **kwargs):
        return self.inidices

    def _post_default(self, list_of_res, *args, **kwargs):
        ...
        return self

    @action
    @inbatch_parallel(init='_init_default', post='_post_default')
    def some_action(self, item_id)
        # process an item and return a value for that item
        return proc_value
```
Here `_post_default` will be called as
```python
_post_default([proc_value_from_1, proc_value_from_2, ..., proc_value_from_last])
```

If anything went wrong than instead of `proc_value`, there would be an instance of some Exception or Error caught in the parallelized function.

This is where `any_action_failed` might come in handy:
```python
from dataset import Batch, action, inbatch_parallel, any_action_failed

class MyBatch(Batch):
    ...
    def _post_default(self, list_of_res, *args, **kwargs):
        if any_action_failed(list_of_res):
            # something went wrong
        else:
            # process the results
        return self

    @action
    @inbatch_parallel(init='_init_default', post='_post_default')
    def some_action(self, item_id)
        # process an item and return a value for that item
        return proc_value
```

`Post`-function should return an instance of a batch class (not necessarily the same). Most of the time it would be just `self`.


## Targets
There are four targets available: `threads`, `nogil`, `async`, `mpc`

### threads
A method will be parallelized with [concurrent.futures.ThreadPoolExecutor](https://docs.python.org/3/library/concurrent.futures.html#threadpoolexecutor).
Take into account that due to [GIL](https://wiki.python.org/moin/GlobalInterpreterLock) only one python thread will run in any given moment (pseudo-parallelism).
However, for a function with intesive I/O processing or waiting for some synchronization you might get a considerable performance increase.

This is a default engine which is used if you don't specify `target` in the `inbatch_parallel` decorator.

### nogil
To get rid of GIL you might write a [cython](http://cython.org/) or [numba](http://numba.pydata.org/) function which will run in parallel.
And a decorated method should just return this nogil-function which will be further parallelized.

```python
from numba import njit
from dataset import Batch, action, inbatch_parallel

@njit(nogil=True)
def numba_fn(data, index, arg):
    # do something
    return new_data

class MyBatch(Batch):
    ...
    @action
    @inbatch_parallel(init='_init_default', post='_post_default', target='nogil')
    def some_action(self, arg)
        # do not process anything, just return nogil-function
        return numba_fn
```

### async
For i/o-intensive processing you might want to consider writing an [`async` method](https://docs.python.org/3/library/asyncio-task.html).
```python
class MyBatch(Batch):
    ...
    @action
    @inbatch_parallel(init='_init_default', post='_post_default', target='async')
    async def some_action(self, item, some_arg)
        # do something asynchrounously
        proc_value = await other_async_function(some_arg)
        return proc_value
```

### mpc
With `mpc` you might run calculations in separate processes thus removing GIL restrictions. For this [concurrent.futures.ProcessPoolExecutor](https://docs.python.org/3/library/concurrent.futures.html#processpoolexecutor) will be used. Likewise `nogil` the decorated method should just return a function which will be executed in a separate process.

```python
from dataset import Batch, action, inbatch_parallel

def mpc_fn(data, index, arg):
    # do something
    return new_data

class MyBatch(Batch):
    ...
    @action
    @inbatch_parallel(init='_init_default', post='_post_default', target='mpc')
    def some_action(self, arg)
        # do not process anything, just return a function which will be run as a separate process
        return mpc_fn
```
Multiprocessing requires all code and data to be serialized (with [pickle](https://docs.python.org/3/library/pickle.html)) in order to be sent to another process. And many classes and methods are not so easy (or even impossible) to pickle. That is why we chose to parallelize functions. And with these thoughts in mind you should carefully consider your parallelized function and the arguments it receives.

Besides, you might want to implement a thorough logging mechanism as multiprocessing configurations are susceptible to hanging up. Without logging it would be quite hard to understand what happened and then debug your code. 


## Arguments with default values
If you have a function with default arguments, you may call it without passing those arguments. 
```python
class MyBatch(Batch):
    ...
    @action
    @inbatch_parallel(init='_init_default', post='_post_default', target='mpc')
    def some_action(self, arg1, arg2, arg3=3)
        ...

# arg3 takes the default value = 3
batch.some_action(1, 2)
```
However, when you call it this way, the default arguments are not available externally (in particular, in decorators).
This is the problem for `nogil`/`mpc` parallelism. 

The best solutions would be not to use default values at all, but if you really need them you should copy them into parallelized functions:
```python
def mpc_fn(item, arg1, arg2, arg3=10):
    # ...

class MyBatch(Batch):
    ...
    @action
    @inbatch_parallel(init='_init_default', post='_post_default', target='mpc')
    def some_action(self, arg1, arg2, arg3=10)
        return mpc_fn
```

You might also return a [partial](https://docs.python.org/3/library/functools.html#functools.partial) with these arguments:
```python
from functools import

def mpc_fn(item, arg1, arg2, arg3=10):
    # ...

class MyBatch(Batch):
    ...
    @action
    @inbatch_parallel(init='_init_default', post='_post_default', target='mpc')
    def some_action(self, arg1, arg2, arg3=10)
        return partial(mpc_fn, arg3=arg3)
```

## Examples
...
