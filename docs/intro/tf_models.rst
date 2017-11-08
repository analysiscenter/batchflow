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
            x = conv_block(dim, x, [32, 64, num_classes], 3, layout='cna cna cnaP', strides=2,
                           name='my_network', training=self.is_training)
            x = tf.identity(x, name='predictions')

Note that you can use this model for 1d, 2d and 3d inputs (with a proper config when initializing a model).

Also take a look into `conv_block <tf_layers#convolution-block>`_ documentation to find out how to write complex networks
in just one line of code and other sophisticated examples.

Now you can train the model in a simple pipeline:

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
