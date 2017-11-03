Tensorflow models
=================

Configuration
-------------

.. autoclass:: dataset.models.tf.TFModel
    :noindex:


How to configure model inputs
-----------------------------
.. automethod:: dataset.models.tf.TFModel._make_inputs
    :noindex:


How to write a custom model
---------------------------

Usually, the only thing you need is to redefine ``_build()`` method.

.. automethod:: dataset.models.tf.TFModel._build
    :noindex:


Example
-------

.. code-block:: python

    from dataset.models.tf import TFModel
    from dataset.models.tf.layers import conv_block, global_average_pooling

    class MyModel(TFModel):
        def _build(self):
            names = ['images', 'labels']
            placeholders, inputs = self._make_inputs(names)

            # a number of dimensions may be defined in a model config
            # default is 2d
            dim = self.get_from_config('dim', 2)
            num_classes = self.num_classes('labels')

            x = inputs['images']
            x = conv_block(dim, x, 32, 3, layout='cnap', name='conv1', training=self.is_training)
            x = conv_block(dim, x, 64, 3, layout='cnap', name='conv2', training=self.is_training)
            x = conv_block(dim, x, num_classes, 3, layout='cnap', name='conv3', training=self.is_training)
            x = global_average_pooling(dim, x, name='predictions')

Note that you can use this model for 1d, 2d and 3d inputs (with a proper config when initializing a model).

Now you can train the model in a pipeline:

.. code-block:: python

    config = {
        'loss': 'ce',
        'decay': 'invtime',
        'optimizer': 'Adam',
        'dim': 2,
        'inputs': dict(images={'shape': (128, 128, 3)},
                       labels={'shape': 10, 'transform': 'ohe', 'name': 'targets'})
    }

    pipeline = my_dataset.p
        .init_variable('loss_history', init_on_each_run=list)
        .init_model('dynamic', MyModel, 'my_model', config)
        .train_model('my_model', fetches='loss',
                     feed_dict={'images': B('images'),
                                'labels': B('labels')},
                     save_to=V('loss_history'), mode='a')
        .run(BATCH_SIZE, shuffle=True, n_epochs=5)
