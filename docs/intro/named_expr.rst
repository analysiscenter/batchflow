=================
Named expressions
=================
As pipelines are declarative you might need a way to address data which exists only when the pipeline is executed.
For instance, model training takes batch data, but you don't have any batches when you declare a pipeline.
Batches appear only when the pipeline is run. This is where **named expressions** come into play.

A named expression specifies a substitution rule. When the pipeline is being executed,
a named expression is replaced with the value calculated according to that rule.

There are several types of named expressions:

* B('name') - a batch class attribute or component name
* V('name') - a pipeline variable name
* C('name') - a pipeline config option
* F(...) - a callable which takes a batch (could be a batch class method or an arbitrary function)
* L(...) - an arbitrary callable (the current batch won't be passed as a parameter)
* R(...) - a random value
* W(...) - a wrapper for a named expression
* P(...) - a wrapper for parallel actions


B - batch component
===================
::

    pipeline
        ...
        .train_model(model_name, feed_dict={'features': B('features'), 'labels': B('labels')})
        ...

At each iteration ``B('features')`` and ``B('labels')`` will be replaced with ``current_batch.features``
and ``current_batch.labels``, i.e. `batch components <components>`_ or attributes.

.. note:: ``B()`` without name returns the batch itself.
          To avoid unexpected changes of the batch, the copy can be created with ``B(copy=True)``.


V - pipeline variable
=====================
::

    pipeline
        ...
        .train_model(V('model_name'), ...)
        ...

At each iteration ``V('model_name')`` will be replaced with the current value of ``pipeline.get_variable('model_name')``.


C - config option
=================
::

    config = dict(model=ResNet18, model_config=model_config)

    train_pipeline = dataset.train.pipeline(config)
        ...
        .init_model('dynamic', C('model'), 'my_model', C('model_config'))
        ...

At each iteration ``C('model')`` will be replaced with the current value of ``pipeline.config['model']``.

This is an example of a model independent pipeline which allows to change models, for instance,
to assess performance of various models.


F - callable
============
A function which takes a batch and, possibly, other arguments.

It can be a lambda function::

    pipeline
        .init_model('dynamic', MyModel, 'my_model', config={
            'inputs': {'images': {'shape': F(lambda batch: batch.images.shape[1:])}}
        })

or a batch class method::

    pipeline
        .train_model(model_name, make_data=F(MyBatch.pack_to_feed_dict, task='segmentation'))

or a function::

    def get_boxes(batch, shape):
        x_coords = slice(0, shape[0])
        y_coords = slice(0, shape[1])
        return batch.images[:, y_coords, x_coords]

    pipeline
        ...
        .update_variable(var_name, F(get_boxes, V('image_shape')))
        ...

or any other Python callable.


.. note:: Most of the time the first parameter passed to ``F``-function contains the current batch.
   However, there are a few exceptions.

As static models are initialized before a pipeline is run (i.e. before any batch is created),
all ``F``-functions specified in static ``init_model`` get ``pipeline`` as the first parameter.

In ``train_model`` and ``predict_model`` ``F``-functions take the batch as the first parameter and the model
as the second parameter. So you can adapt the function to specific models.


L - callable
============
A function which takes arbitrary arguments.::

    pipeline
        ...
        .init_variable('logfile', L(open, 'file.log', 'w'))
        ...


R - random value
================
A sample from a random distribution. All `numpy distributions <https://docs.scipy.org/doc/numpy/reference/routines.random.html#distributions>`_ are supported::

    pipeline
        .some_action(R('uniform'))
        .other_action(R('beta', 1, 1, seed=14))
        .yet_other_action(R('poisson', lam=4, size=(2, 5)))
        .one_more_action(R(['opera', 'ballet', 'musical'], p=[.1, .15, .75], size=15, seed=42))


W - a wrapper
=============
To pass a named expression to an action without evaluating it within a pipeline you can wrap it::

    pipeline
        .some_action(arg=W(V('variable'))

As a result ``some_action`` will get not a current value of a pipeline variable, but a ``V``-expression itself.


P - a parallel wrapper
======================
It comes in handy for parallel actions so that :doc:`@inbatch_parallel <parallel>` could determine that
different values should be passed to parallel invocations of the action.

.. note:: For ``R``-expressions the default ``size`` will be ``B('size')``.

For instance, each item in the batch will be rotated at its own angle::

    pipeline
        .rotate(angle=P(R('uniform', -30, 30)))

Without ``P`` all images in the batch will be rotated at the same angle,
since an angle randomized across batches only::

    pipeline
        .rotate(angle=R('normal', 0, 1))

Every image in the batch gets a noise of the same intensity (7%), but of a different color::

    pipeline.
        .add_color_noise(p_noise=.07, color=P(R('uniform', 0, 255, size=10)))

.. note:: If a batch size is greater than the variable value size, than an exception will be raised
          as there is not enough values for each parallel invocations of an action.

``P`` can be used not only with ``R``-expressions::

    pipeline
        .some_action(P(V('loss_history')))
        .other_action(P(C('model_class')))
        .yet_other_action(P(B('size')))
