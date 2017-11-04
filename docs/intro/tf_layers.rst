Tensorflow layers and losses
============================

Convolution block
-----------------
The module :mod:`.models.tf.layers` includes a :func:`convolution building block <.models.tf.layers.conv_block>`
which helps build complex networks in a concise way.
The block consist of predefined layers, among which:

- convolutions (as well as dilated and separable convolutions)
- transposed convolutions
- batch normalization
- activation
- max pooling
- global max pooling
- average pooling
- global average pooling
- maximum intensity projection (mip)
- dropout

The layers types and order are set by ``layout`` parameter. Thus, for instance, you can easily create
a sequence of 4 layers (3x3 convolution, batch norm, relu and max pooling) in one line of code:

.. code-block:: python

    x = conv2d_block(dim, x, 32, 3, layout='cnap', name='conv1', training=self.is_training)


Or a more sophisticated example - a full 14-layer VGG-like model in just 6 lines:

.. code-block:: python

    class MyModel(TFModel):
        def _build(self):
            placeholders, inputs = self._make_inputs()

            dim = 2
            num_classes = self.num_classes('labels')

            x = inputs['images']
            x = conv_block(dim, x, 64, 3, layout='cacap', name='block1')
            x = conv_block(dim, x, 128, 3, layout='cacap', name='block2')
            x = conv_block(dim, x, 256, 3, layout='cacacap', name='block3')
            x = conv_block(dim, x, 512, 3, layout='cacacap', name='block4')
            x = conv_block(dim, x, 512, 3, layout='cacacap', name='block5')
            x = conv_block(dim, x, num_classes, 3, layout='caP', name='classification')
            output = tf.identity(x, name='predictions')

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

When ``layout`` includes several layers of the same type, each one can have its own parameters,
if corresponding arguments are passed as lists/tuples.

A canonical bottleneck block (1x1, 3x3, 1x1 conv with relu in-between)::

    x = conv_block(2, x, [64, 64, 256], [1, 3, 1], layout='cacac')

A complex Nd block:

- 5x5 conv with 32 filters
- relu
- 3x3 conv with 32 filters
- relu
- 3x3 conv with 64 filters and a spatial stride 2
- relu
- batch norm
- dropout with rate 0.15

::

    x = conv_block(dim, x, [32, 32, 64], [5, 3, 3], layout='cacacand', strides=[1, 1, 2], dropout_rate=.15)

Or the earlier defined 14-layers VGG network as a one-liner::

    x = conv_block(dim, x, [64]*2 + [128]*2 + [256]*3 + [512]*6 + [num_classes], 3, layout='cacap cacap cacacap cacacap cacacap caP')

However, in terms of training performance and prediction accuracy the following block with strided separable (grouped) convolutions
and dropout will perform a way better::

        x = conv_block(dim, x, [16, 32, 64, num_classes], 3, strides=[2, 2, 2, 1], dropout_rate=.15,
                       layout='cna cna cna cnaP', depth_multiplier=[1, 2, 2, 1], training=self.is_training)

1d transposed convolution
-------------------------

.. autofunction:: dataset.models.tf.layers.conv1d_transpose
    :noindex:


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
