Tensorflow layers and losses
============================

Convolution block
-----------------
The module :mod:`.models.tf.layers` includes a :func:`convolution building block <.models.tf.layers.conv_block>`
which helps build complex networks in a concise way.
The block consist of predefined layers, among which:

- convolution
- transposed convolution
- batch normalization
- activation
- max pooling
- dropout

The layers types and order are set by ``layout`` parameter. Thus, for instance, you can easily create
a sequence of 4 layers (3x3 convolution, batch norm, relu and max pooling) in one line of code:

.. code-block:: python

    x = conv2d_block(dim, x, 32, 3, layout='cnap', name='conv1', training=self.is_training)


Or a more sophisticated example - a full 12-layer VGG-like model in just 4 lines:

.. code-block:: python

    class MyModel(TFModel):
        def _build(self):
            placeholders, inputs = self._make_inputs()
            dim = 2
            x1 = conv_block(dim, inputs['images'], 16, 3, layout='cccnap', name='conv1', training=self.is_training)
            x2 = conv_block(dim, x1, 32, 3, layout='cccnap', name='conv2', training=self.is_training)
            x3 = conv_block(dim, x2, 64, 3, layout='cccnap', name='conv3', training=self.is_training)
            x4 = conv_block(dim, x3, 128, 3, layout='cccnap', name='conv4', training=self.is_training)
            output = global_average_pooling(dim, x4, name='predictions')

That's a fully working example. Just try it with a simple pipeline:

.. code-block:: python

    from dataset.openses import MNIST
    from dataset.models.tf import TFModel
    from dataset.models.tf.layers import conv_block, global_average_pooling

    mnist = MNIST()

    train_pp = (mnist.train.p
                .init_variable('current_lost', 0)
                .init_model('dynamic', MyModel, 'conv',
                            config={'loss': 'ce',
                                    'inputs': dict(images={'shape': (28, 28, 1)},
                                                   labels={'shape': 10, 'dtype': 'uint8',
                                                           'transform': 'ohe', 'name': 'targets'})})
                .train_model('conv', fetches='loss', feed_dict={'images': B('images'), 'labels': B('labels')},
                             save_to=V('current_loss'), mode='a')
                .print_variable('current_loss')
                .run(128, shuffle=True, n_epochs=2))


1d transposed convolution
-------------------------

.. autofunction:: dataset.models.tf.layers.conv1d_transpose


Pooling
-------

.. autofunction:: dataset.models.tf.layers.max_pooling
    :noindex:

.. autofunction:: dataset.models.tf.layers.average_pooling
    :noindex:

.. autofunction:: dataset.models.tf.layers.global_max_pooling
    :noindex:

.. autofunction:: dataset.models.tf.layers.global_average_pooling
    :noindex:

.. autofunction:: dataset.models.tf.layers.mip
    :noindex:

Flatten
-------
.. autofunction:: dataset.models.tf.layers.flatten
    :noindex:

.. autofunction:: dataset.models.tf.layers.flatten2d
    :noindex:
