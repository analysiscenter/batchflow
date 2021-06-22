========
Pipeline
========


Introduction
============

Quite often you can't just use the data itself, as it needs some specific preprocessing beforehand. And not too rarely
you end up with several processing workflows which you have to use simultaneously.
That is the situation when pipelines might come in handy.

Firstly, you create a batch class with all necessary actions::

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
       def yet_another_action(self):
           ...
           return self

Secondly, you create a dataset (`client_index` is an instance of :ref:`DatasetIndex`)::

   ct_ds = Dataset(client_index, batch_class=ClientTranasactions)

And now you can define a workflow pipeline::

   trans_pipeline = (ct_ds.pipeline()
                       .some_action()
                       .other_action(param=2)
                       .yet_another_action())

And nothing happens! Because all the actions are lazy.
Let's run them.::

   trans_pipeline.run(BATCH_SIZE, shuffle=False, n_epochs=1)

Now the dataset is split into batches and then all the actions are executed for each batch independently.

In the very same way you can define an augmentation workflow::

   augm_wf = (image_dataset.pipeline()
               .load('/some/path')
               .random_rotate(angle=(-30, 30))
               .random_resize(factor=(0.8, 1.2))
               .random_crop(factor=(0.5, 0.8))
               .resize(shape=(256, 256))
   )

And again, no action is executed until its result is needed.::

   NUM_ITERS = 1000
   for i in range(NUM_ITERS):
       image_batch = augm_wf.next_batch(BATCH_SIZE, shuffle=True, n_epochs=None)
       # only now the actions are fired and data is changed with the workflow defined earlier


Actions
=======

Pipeline actions might come from 3 sources:

- Pipeline API
- batch class actions
- arbitrary namespaces.

:doc:`Pipeline API <../api/batchflow.pipeline>` contains operations with variables and models.

:doc:`Batch class <batch>` comprises data loading operations, preprocessing methods and augmentations.
A batch class method marked with :ref:`action <actions>` decorator might be used in a pipeline workflow.

Besides, pipeline actions chains might include arbitrary functions from given namespaces.


Actions from namespaces
-----------------------
Just add a namespace (e.g. a class, a module) which contains the functions needed within a pipeline.

::

   pipeline = (dataset.pipeline()
                .add_namespace(numpy, mymodule)    # numpy and mymodule methods are now accessible within the pipeline
                .init_variable("var")              # Pipeline API
                .init_model("resnet", ResNet18)    # Pipeline API
                .resize((128, 128))                # batch class API, namely ImagesBatch
                .my_func(10, save_to=V("var"))     # call a function from mymodule and store its result into a pipeline variable
                .print(V("var"))                   # Pipeline API again

The result of these actions can be stored with `save_to` parameter::

    pipeline.before
        ...
        .some_func(save_to=V('some_var')))

Normally, `named expressions <named_expr>` are used in `save_to`. However, lists or numpy arrays also work out.
Note that `save_to` argument is never passed to a function, since it is fully processed within the pipeline.

For convenience the main module and the dataset class are automatically added to namespaces available.
So you can use dataset methods or functions defined right in the main module in the pipeline chain.

::

    class MyDataset(Dataset):
        def dataset_method(self):
            print("dataset method")
            return 200

    def main_func():
        print("main func")
        return 100

    pipeline.
        .dataset_method(save_to=V('return_from_method'))
        .main_func(save_to=V('return_from_func'))


.. _after_pipeline:

Before and after pipelines
==========================
More complicated pipelines include setup and tear down actions. That's exactly what `before` and `after` pipelines are supposed to do.

::

    pipeline.before
            .add_namespace(mymodule)           # mymodule methods are now accessible within the pipeline
            .init_variable("var")              # Pipeline API
            .init_model("my-model", ResNet18)   # Pipeline API
            .connect_to_mydb(USER, PASSWORD)   # a method from mymodule

    pipeline.after
        .add_namespace(mymodule)           # mymodule methods are now accessible within the pipeline
        .save_model("ResNet18", path='/some/path')     #Pipeline API
        .disconnect_from_mydb()            # a method from mymodule

`before` and `after` pipelines are executed automatically when the main pipeline is executed (specifically, before and after it).

Note that the main module and the dataset class are automatically added to namespaces available.
And result of these actions can be retrieved and stored with `save_to` parameter.

::

    class MyDataset(Dataset):
        def dataset_method(self):
            print("dataset method")
            return 200

    def main_func():
        print("main func")
        return 100

    pipeline.before
        .dataset_method(save_to=V('return_from_method'))
        .main_func(save_to=V('return_from_func'))

However, take into account that when you iterate over the pipeline with `gen_batch(...)` or `next_batch(...)`, `after`-pipeline
will be executed automatically only when the iteration is fully finished.
If you break the iteration process (e.g. when early stopping is occurred or when exception is caught),
you should explicitly call `pipeline.after.run()`.

See :doc:`API <../api/batchflow.once_pipeline>` for methods available in `before` and `after` pipelines.

The whole pipeline can be chained in a single instruction::

    pipeline
        .before
            .add_namespace(mymodule)           # mymodule methods are now accessible within the pipeline
            .init_variable("var")              # Pipeline API
            .init_model("my-model", ResNet18)   # Pipeline API
            .connect_to_mydb(USER, PASSWORD)   # a method from mymodule

        .main
            load(dst='features')
            train_model('my-model', B.features)

        .after
            .add_namespace(mymodule)           # mymodule methods are now accessible within the pipeline
            .save_model("ResNet18", path='/some/path')     #Pipeline API
            .disconnect_from_mydb()            # a method from mymodule

    pipeline.run(BATCH_SIZE)


Algebra of pipelines
====================

There are two ways to define a pipeline:

* a chain of actions
* a pipeline algebra

An action chain is a concise and convenient way to write pipelines. But sometimes it's not enough, for instance, when you want to manipulate with many pipelines adding them or multiplying as if they were numbers or matrices. And that's what we call `a pipeline algebra`.

There are 5 operations available: `+`, `*`, `@`, `<<`, `>>`.

concat `+`
----------
Add two pipelines by concatenating them, so the actions from the first pipeline will be executed before actions from the second one.
`p.resize(shape=(256, 256)) + p.rotate(angle=45)`

repeat `*`
----------
Repeat the pipeline several times.
`p.random_rotate(angle=(-30, 30)) * 3`

sometimes `@`
-------------
Execute the pipeline with the given probability.
`p.random_rotate(angle=(-30, 30)) @ 0.5`

`>>` and `<<`
-------------
Link a pipeline to a dataset.
`dataset >> pipeline` or `pipeline << dataset`

Or update pipeline's config.
`config >> pipeline` or `pipeline << config`

The complete example::

   from batchflow import Pipeline

   with Pipeline() as p:
       preprocessing_pipeline = p.load('/some/path')
                                + p.resize(shape=(256, 256))
                                + p.random_rotate(angle=(-30, 30)) @ .8
                                + p.random_transform() * 3
                                + p.random_crop(shape=(128, 128))

   images_prepocessing = preprocessing_pipeline << images_dataset


Creating pipelines
==================

Pipelines can be created from scratch or from a dataset.

A template pipeline
-------------------

The code below creates a pipeline from scratch.

.. code-block:: python

   from batchflow import Pipeline

   my_pipeline = Pipeline()
                   .some_action()
                   .another_action()

Or one can use a context manager with pipeline algebra::

   from batchflow import Pipeline

   with Pipeline() as p:
       my_pipeline = p.some_action() + p.another_action()

However, you cannot execute this pipeline as it doesn't linked to any dataset.
On the other hand, such pipelines might be applied to different datasets::

   cifar10_pipeline = template_preprocessing_pipeline << cifar10_dataset
   mnist_pipeline = template_preprocessing_pipeline << mnist_dataset

A dataset pipeline
------------------

::

   my_pipeline = my_dataset.pipeline()
                   .some_action()
                   .another_action()

Or a shorter version::

   my_pipeline = my_dataset.p
                   .some_action()
                   .another_action()

Every call to `dataset.pipeline()` or `dataset.p` creates a new pipeline.


Running pipelines
=================

There are 5 ways to execute a pipeline.

Batch generator
---------------

:meth:`~.Pipeline.gen_batch`::

   for batch in my_pipeline.gen_batch(BATCH_SIZE, shuffle=True, n_epochs=2, drop_last=True):
       # do whatever you want

`batch` will be the batch returned from the very last action of the pipeline.

.. note:: `BATCH_SIZE` is a size of the batch taken from the dataset. Actions might change the size of the batch and thus the batch you will get from the pipeline might have a different size.

.. note:: Pipeline execution might take a long time so showing a progress bar might be helpful. Just add `bar=True` to gen_batch parameters.

To start from scratch, `reset` parameter can be used::

    for batch in my_pipeline.gen_batch(BATCH_SIZE, shuffle=True, n_epochs=2, drop_last=True, reset='vars'):
       # do whatever you want

You might reset pipeline variables, the batch iterator and models. See :meth:`~.Pipeline.reset` for details.


Run
---

To execute the pipeline right now for all iterations at once call :meth:`~.Pipeline.run`::

   my_pipeline = (dataset.p
      .some_action()
      .other_action()
      .yet_another_action()
      .run(BATCH_SIZE, n_epochs=2, drop_last=True, bar=True)
   )

Some people prefer a slightly longer, but a bit more certain name :meth:`~.Pipeline.run_now`.

Usually `run` is used to execute the pipeline without resetting all the variables and models
(thus continuing the execution which started earlier and keeping the values and trained models).
However, you might want to start from scratch re-initialzing the variables and/or the models::

    my_pipeline.run_now(BATCH_SIZE, n_iters=1000, reset='variables')

or::

    my_pipeline.run_now(BATCH_SIZE, n_iters=1000, reset='models')

or even:

    my_pipeline.run_now(BATCH_SIZE, n_iters=1000, reset='all')

In this case the pipeline variables will be reinitialized and the modes will be reset to initial untrained state.


Lazy run
--------

You can add `run` with `lazy=True` or just :meth:`~.Pipeline.run_later` as the last action in the pipeline and
then call `run()` or `next_batch()` without arguments at all::

    my_pipeline = (dataset.p
        .some_action()
        .other_action()
        .yet_another_action()
        .run_later(BATCH_SIZE, n_epochs=None, drop_last=True)
    )

    for i in range(MAX_ITER):
        batch = my_pipeline.next_batch()
        # do whatever you want


next_batch function
-------------------

:meth:`~.Pipeline.next_batch`::

   for i in range(MAX_ITER):
       batch = my_pipeline.next_batch(BATCH_SIZE, shuffle=True, n_iters=1000, drop_last=True)
       # do whatever you want

If you need to start from scratch, you might call :meth:`~.Pipeline.reset` beforehand::

    my_pipeline.reset('models')
    my_pipeline.reset('variables')
    my_pipeline.reset('all')

Or pass `reset` parameter to `next_batch`.


Single execution
----------------
A pipeline might be run for one given batch only with :meth:`~.Pipeline.execute_for`::

    res_batch = my_pipeline.execute_for(batch)



Pipeline variables
==================

Sometimes batches can be processed in a "do and forget" manner: when you take a batch, make some data transformations and then switch to another batch.
However, not infrequently you might need to remember some parameters or intermediate results (e.g. a value of loss function or accuracy on every batch
to draw a graph later). This is why you might need pipeline variables.

Initializing a variable
-----------------------

.. code-block:: python

    my_pipeline = (my_dataset.p
        .init_variable("my_variable", 100)
        .init_variable("some_counter", init_on_each_run=0)
        .init_variable("var with init function", F(my_init_function))
        .init_variable("loss_history", init_on_each_run=list)
        .first_action()
        .second_action()
        ...
    )

To initialize a variable just add to a pipeline `init_variable(...)` with a variable name and a default value.
Variables might be initialized once in a lifetime (e.g. some global state or a configuration parameter) or before each run
(like counters or history stores).

Sometimes it is more convenient to initialize variables indirectly through a function. For instance, `loss_history` cannot be initialized with `[]`
as it would make a global variable which won't be cleared on every run. What you actually need is to call `list()` on each run.

Init functions are also a good place for some complex logic or randomization.

Updating a variable
-------------------

Each batch instance have a pointer to the pipeline it was created in (or `None` if the batch was created manually).

So getting an access to a variable is easy::

    class MyBatch(Batch):
        ...
        @action
        def some_action(self):
            var_value = self.pipeline.get_variable("variable_name")
            ...

If a variable does not exist, it might be created and initialized, if `create` parameter is set to `True`.
For a flexible initialization `default`, `init` and `init_on_each_run` might also be passed to `get_variable()`.

.. note:: An explicit variable initialization in a pipeline is a preferred way to create variables.

If `create` is `False` (which is by default), then `get_variable` will raise a `KeyError` if a variable does not exist.

`v()` is a shorter alias for `get_variable()`::

    pipeline.v('var_name')

To change a variable value just call `set_variable` within an action::

    class MyBatch(Batch):
        ...
        @action
        def some_action(self):
            ...
            self.pipeline.set_variable("variable_name", new_value)
            ...

Or add `update` to the pipeline::

    my_pipeline
        ...
        .update(V("current_batch_labels"), F(MyBatch.get_labels))
        .update(V("all_labels", mode='append'), V('current_batch_labels'))

The first parameter specifies :doc:`a named expression <named_expr>` where the value will be stored.
The second parameter is an updating value and it can be a value of any type or :doc:`a named expression <named_expr>`.

Note that a named expression might have a mode (e.g. `V('name', mode='a')`) which could be one of:

* `'w'` or `'write'` to rewrite a variable with a new value. This is a default mode.
* `'a'` or `'append'` to append a value to a variable (e.g. if a variable is a list).
* `'e'` or `'extend'` to extend a variable with a new value (e.g. if a variable is a list and a value is a list too).
* `'u'` or `'update'` to update a variable with a new value (e.g. if a variable is a dict).

For sets and dicts `'u'` and `'a'` do the same.

Deleting a variable
-------------------

Just call `pipeline.delete_variable("variable_name")` or `pipeline.del_variable("variable_name")`.

Deleting all variables
----------------------

As simple as `pipeline.delete_all_variables()`.


Update
======

A pipeline might need custom calculations which can be implemented with :meth:`~.Pipeline.save_to` :meth:`~.Pipeline.update`::

    pipeline
        .init_variable('counter', 0)
        ...
        .update(V('counter'), V('counter') + 1)

The first parameter is a named expression where the result will be stored, while the second parameter is an expression
which value will be re-calculated at each iteration.

Some other useful examples might include:

- collecting loss history::

    pipeline
        .init_variable('loss_history', list)
        ...
        .save_to(V('list', mode='a'), V('current_loss'))

- collecting predictions::

    pipeline
        .init_variable('all_predictions', list)
        ...
        .update(V('all_predictions', mode='e'), V('predictions'))

- assessing performance::

    pipeline
        .update(B('time'), L(time.time))
        ...
        .save_to(B('time'), L(time.time) - B('time'))
        .update(V('throughput'), B('images').nbytes / B('time'))
        ...

The methods are synonyms and might be used interchangably to fit your pipeline narrative.


Pipeline locks
==============

If you use multi-threading :doc:`prefetching <prefetch>` or :doc:`in-batch parallelism <parallel>`,
than you might require synchronization when accessing some shared resource::

    dataset.p
        ...
        .init_lock('lock_name')
        ... # common section
        .acquire_lock('lock_name')
        ... # a critical section
        .release_lock('lock_name')
        ...
        .run(BATCH_SIZE, prefetch=PARALLEL_BATCHES)

Locks are stored as variables with the very same names. So you can access a lock easiliy as ``pipeline.v('lock_name')``
and use it within actions or even custom functions.

``init_lock`` is not required as the first acquire will create the lock if needed.
However, ``init_lock`` allow for cleaner and tidier pipelines.


Join and merge
==============

Joining pipelines
-----------------

If you have a pipeline `images` and a pipeline `labels`, you might join them for a more convenient processing::

    images_with_labels = (images.p
        .load(...)
        .resize(shape=(256, 256))
        .random_rotate(angle=(-pi/4, pi/4))
        .join(labels)
        .some_action()
    )

When this pipeline is run, the following will happen for each batch of `images`:

* the actions `load`, `resize` and `random_rotate` will be executed
* a batch of `labels` with the same index will be created
* the `labels` batch will be passed into `some_action` as a first argument (after `self`, of course).

So, images batch class should look as follows::

   class ImagesBatch(Batch):
       def load(self, src, fmt):
           ...

       def resize(self, shape):
           ...

       def random_rotate(self, angle):
           ...

       def some_actions(self, labels_batch):
           ...

You can join several sources::

    full_images = (images.p
        .load(...)
        .resize(shape=(256, 256))
        .random_rotate(angle=(-30, 30))
        .join(labels, masks)
        .some_action()
    )

Thus, the tuple of batches from `labels` and `masks` will be passed into `some_action` as the first arguments (as always, after `self`).

Mostly, `join` is used as follows::

    full_images = (images.p
        .load(...)
        .resize(shape=(256, 256))
        .join(labels, masks)
        .load(components=['labels', 'masks'])
    )

See :func:`~batchflow.Batch.load` for more details.

Merging pipelines
-----------------

You can also merge data from two pipelines (this is not the same as `concatenating pipelines <#algebra-of-pipelines>`_).::

    images_with_augmentation = (images_dataset.p
        .load(...)
        .resize(shape=(256, 256))
        .random_rotate(angle=(-30, 30))
        .random_crop(shape=(128, 128))
        .run(batch_size=16, epochs=None, shuffle=True, drop_last=True, lazy=True)
    )

    all_images = (images_dataset.p
        .load(...)
        .resize(shape=(128, 128))
        .merge(images_with_augmentation)
        .run(batch_size=16, epochs=3, shuffle=True, drop_last=True)
    )

What will happen here is

* `images_with_augmentation` will generate batches of size 16
* `all_images` before merge will generate batches of size 16
* `merge` will combine both batches in some way.

Pipeline's `merge` calls `batch_class.merge([batche_from_pipe1, batch_from_pipe2])`.

The default `Batch.merge` just concatenate data from both batches, thus making a batch of double size.

Take into account that the default `merge` also changes index to `numpy.arange(new_size)`.


Rebatch
=======

When actions change the batch size (for instance, dropping some bad or skipping incomplete data),
you might end up in a situation when you don't know the batch size and, what is sometimes much worse,
batch size differs. To solve this problem, just call `rebatch`::

    images_pipeline = (images_dataset.p
        .load(...)
        .random_rotate(angle=(-30, 30))
        .skip_black_images()
        .skip_too_noisy_images()
        .rebatch(32)
    )

Under the hood `rebatch` calls `merge`, so you must ensure that `merge` works properly for your specific data and write your own `merge` if needed.


Exceptions
==========

SkipBatchException
------------------
Sometimes you might want to stop processing a batch within a pipeline (e.g. if it does not meet a certain criteria or contains erroneous data, etc).
Just throw :class:`~batchflow.SkipBatchException` in an action method and the pipeline will skip this batch and switch to a new one.


EmptyBatchSequence
------------------
When you call several `run` (or `gen_batch`, or `next_batch`) one after another without resetting a batch iterator (see :meth:`~batchflow.Pipeline.reset`),
you might bump into a situation when the batch iterator is exhausted. Whenever this happens, :class:`~batchflow.EmptyBatchSequence` warning is emitted,
which can ba caught if needed::

    try:
        pipeline.run()
    except EmptyBatchSequence:
        print("There are no batches left. Call pipeline.reset('iter').")



Models
======
See :doc:`Working with models <models>`.


Debugging
=========
To debug a pipeline you might pass `debug` parameter when executing the pipeline, e.g.::

    pipeline.run(batch_size=100, debug=True)

Execution parameters are gathered into `pipeline.debug_info` data frame, which contains
batch id, action name, start time and execution time.

This might help you to analyze pipeline exection, find bottlenecks and so on.


API
===
See :doc:`pipelines API <../api/batchflow.pipeline>`.
