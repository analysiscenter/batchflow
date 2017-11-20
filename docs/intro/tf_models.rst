Tensorflow models
=================


How to write a custom model
---------------------------

To begin with, take a look into `conv_block <tf_layers#convolution-block>`_ documentation to find out how to write
complex networks in just one line of code.


The simplest case you should avoid
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
All you need is just redefine ``body()`` method.
For example, let's create a small fully convolutional network with 3x3 convolutions, batch normalization, dropout
and a global average pooling at the end::

    from dataset.models.tf import TFModel
    from dataset.models.tf.layers import conv_block

    class MyModel(TFModel):
        def body(self, **kwargs):
            names = 'images', 'labels'
            config = self.build_config(names)

            inputs = self.inputs['images']
            num_classes = self.num_classes('labels')

            x = conv_block(inputs, filters=[64, 128, num_classes], kernel_size=3,
                           layout='cna cna cna dV', dropout_rate=.2)
            return x

Despite simplicity, this approach is highly discouraged as it prevents configuring the model within a pipeline and
does not allow model composition, i.e. using this network components in other networks.

The right way
~~~~~~~~~~~~~

Here we split network configuration and network definition into separate methods.::

    from dataset.models.tf import TFModel
    from dataset.models.tf.layers import conv_block

    class MyModel(TFModel):
        @classmethod
        def default_config(cls):
            config = TFModel.default_config()
            config['body'].update(dict(filters=[64, 128], kernel_size=3, layout='cna cna'))
            config['head'].update(dict(kernel_size=3, layout='cna dV', dropout_rate=.2))
            return config

        def build_config(self, names=None):
            names = names if names else ['images', 'labels']
            config = super().build_config(names)

            config['common']['data_format'] = self.data_format('images')
            config['input_block']['inputs'] = self.inputs['images']
            config['head']['filters'] = self.num_classes('labels')
            return config

        @classmethod
        def body(cls, inputs, name='body', **kwargs):
            kwargs = cls.fill_params('body', **kwargs)
            x = conv_block(inputs, **kwargs)
            return x

Note that ``default_config`` and ``body`` are now ``@classmethods`` which means that they might be called without
instantiating a ``MyModel`` object.
This is needed for model composition, e.g. ``MyModel`` might serve as a base network for an FCN or SSD network.

However, ``build_config`` is still an ordinary method, so it is called only when an instance of ``MyModel`` is created.

Thus, ``default_config`` should contain all the constants and default values which are totaly independent of the dataset
and a specific task at hand, while ``build_config`` is intended to extract values from dataset through pipeline's initialization variables
(for details see `Configuring a model <models#configuring-a-model>`_ and `TFModel configuration <#configuration>`_ below).


Now you can train the model in a simple pipeline::

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


Configuration
-------------

.. autoclass:: dataset.models.tf.TFModel
    :noindex:


How to configure model inputs
-----------------------------
.. automethod:: dataset.models.tf.TFModel._make_inputs
    :noindex:


