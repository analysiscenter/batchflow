===============================
Batch class for handling images
===============================

ImagesBatch class handles 2D images and their labels.

Components
----------

The class has two components: ``images`` and ``labels``.

Augmentation
------------

ImagesBatch provides typical augmentation actions:

* :meth:`crop <~dataset.ImagesBatch._crop_>` -- crop rectangular area from an image
* :meth:`flip <~dataset.ImagesBatch._flip__all>` -- flip an image (left to right or upside down)
* :meth:`scale <~dataset.ImagesBatch._scale_>` -- scale an image (stretch or tie)
* :meth:`put_on_background <~dataset.ImagesBatch._put_on_background_>` -- put an image on a given background
* :meth:`resize <~dataset.ImagesBatch._resize_>` -- resize an image a to given shape
* :meth:`shift <~dataset.ImagesBatch._shift_>` -- shift an image (cropping pixels that are out of scope)
* :meth:`rotate <~dataset.ImagesBatch._rotate_>` -- rotate an image by angle in degrees
* :meth:`pad <~dataset.ImagesBatch._pad_>` -- add constant values to the border of an image (enlarging the last's shape)
* :meth:`invert <~dataset.ImagesBatch._invert_>` -- invert given channels in an image
* :meth:`salt <~dataset.ImagesBatch._salt_>` -- set pixels in random positions to given colour
* :meth:`multiply <~dataset.ImagesBatch._multiply_>` -- multiply an image by given number
* :meth:`add <~dataset.ImagesBatch._add_>` -- add given term to an image

.. note:: All these methods can be executed for randomly sampled images from a batch. You just need to specify ``p`` while calling an action (probability of applying an action to an image).

.. note:: Use ``R`` or ``S`` `named expressions :ref:<named_expr>` to sample arguments for actions. In the first case argument will be sampled for all images in a batch. If ``S`` is passed then argument will be sampled for each image.

Examples:

All images in a batch are rotated by 10 degrees:
.. code-block:: python

    ...
    (Pipeline().
        ...
        .rotate(angle=10)
        ...

All images in a batch are rotated by the common angle sampled from the normal distribution
.. code-block:: python

    ...
    (Pipeline().
        ...
        .rotate(angle=R('normal', loc=0, scale=1))
        ...

Each image in a batch are rotated by its own sampled angle
.. code-block:: python

    ...
    (Pipeline().
        ...
        .rotate(angle=P(R('normal', loc=0, scale=1)))
        ...


Rotate each image with probability 0.7 by its own sampled angle
.. code-block:: python

    ...
    (Pipeline().
        ...
        .rotate(angle=P(R('normal', loc=0, scale=1)), p=0.7)
        ...


Loading from files
------------------

To load images, use action :meth:`~dataset.BaseImagesBatch.load` with ``fmt='image'``.


Saving
------

To dump images, use action :meth:`~dataset.BaseImagesBatch.dump`


`transform_actions` decorator
-----------------------------

This decorator finds all defined methods whose names starts with user-defined `suffix` and `prefix` then
decorates them with ``wrapper`` which is an argument too.

For example, there are two wrapper functions defined in :class:`~dataset.Batch`:
    1. :meth:`~dataset.Batch.apply_transform_all`
    2. :meth:`~dataset.Batch.apply_transform`

And, by default, all methods that start with '_' and end with '_' are wrapped with the first mentioned method and those ones that start with '_' and end with '_all' are wrapped by the second one.

Defining custom actions
-----------------------

There are 3 ways to define an action:

    1. By writting a classic ``action`` like in  :class:`~dataset.Batch`
    2. By writing a method that takes ``image`` as the first argument and returns transformed one. Method's name must be surrounded by unary '_'.
    3. By writing a method that takes nd.array of ``images`` as the first argument and ``indices`` as the second. This method transforms ``images[indices]`` and returns ``images``. Method's name must start with '_' and end with '_all'.

.. note:: In the last two approaches, actual action's name doesn't include mentioned suffices and prefixes. For example, if you define method ``_method_name_`` then in a pipeline you should call ``method_name``. For more details, see below.

.. note:: Last two methods' names must not be surrounded by double '_' (like `__init__`) otherwise they will be ignored.

Let's take a closer look on the two last approaches:

``_method_name_``
~~~~~~~~~~~~~~~~~

It must have the following signature:

   ``_method_name_(image, ...)``

This method is actually wrapped with :meth:`~dataset.Batch.apply_transform`. And (usually) executed in parallel for each image.


.. note:: If you define these actions in a child class then you must decorate it with ``@transform_actions(prefix='_', suffix='_', wrapper='apply_transform')``

Example:

.. code-block:: python

    @transform_actions(prefix='_', suffix='_', wrapper='apply_transform')
    class MyImagesBatch(ImagesBatch):
        ...
        def _flip_(image, mode):
            """ Flips an image.
            """

            if mode == 'lr':
                image = image[:, ::-1]
            elif mode == 'ud':
                image = image[::-1]
            return image
        ...

To use this action in a pipeline you must write:

.. code-block:: python

    ...
    (Pipeline().
        ...
        .flip(mode='lr')
        ...

.. note:: Note that prefix '_' and suffix '_' are removed from the action's name.

.. note:: All actions written in this way can be applied with given probability to every image. To achieve this, pass parameter ``p`` to an action, like ``flip(mode='lr', p=0.5)``

.. note:: These actions are performed for every image each in its own thread. To change it (for example, execute in asynchronous mode), pass parameter `target` (``.flip(mode='lr', target='a')``). For more detail, see :doc:`<parallel>`.


``_method_name_all``
~~~~~~~~~~~~~~~~~~~~


It must have the following signature:

   ``_method_name_all(images, indices, ...)``

This method is actually wrapped with :meth:`~dataset.Batch.apply_transform_all`. And executed once with the whole batch passed. ``indices`` parameter declares images that must be transformed (it is needed, for example, if you want to perfom action only to the subset of the elemets. See below for more details)


.. note:: If you define these actions in a child class then you must decorate it with ``@transform_actions(prefix='_', suffix='_all', wrapper='apply_transform_all')``

Example:

.. code-block:: python

    @transform_actions_all(prefix='_', suffix='_', wrapper='apply_transform_all')
    def _flip_all(self, images=None, indices=[0], mode='lr'):
        """ Flips images at given indices.
        """

        if mode == 'lr':
            images[indices] = images[indices, :, ::-1]
        elif mode == 'ud':
            images[indices] = images[indices, ::-1]
        return images

To use this action in a pipeline you must write:

.. code-block:: python

    ...
    (Pipeline().
        ...
        .flip(mode='lr')
        ...


.. note:: Note that prefix '_' and suffix '_all' are removed from the action's name.

.. note:: All actions written in this way can be applied with given probability to every image. To achieve this, pass parameter ``p`` to an action, like ``flip(mode='lr', p=0.5)``

.. note:: These actions are performed each in one thread for all batch. Please note that you can't pass ``S`` named expression as an argument (because one transformation is applied to every choozen image).


Assembling after parallel execution
-----------------------------------


To assemble images after parallel execution you can use :meth:`~dataset.ImagesBatch._assemble` method.

.. note:: Note that if images have different shapes after an action then there are two ways to tackle it:
          1. Do nothing. Then images will be stored in `np.ndarray` with `dtype=object`.
          2. Pass `preserve_shape=True` to an action which changes the shape of an image. Then image
            is cropped from the left upper corner (unless action has `origin` parameter, see more in :ref:`Actions`).
