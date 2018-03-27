===========
Research
===========

Research class is intended to multiple running of the same pipelines with different parameters in order to get some metrics value.

Basic usage
-----------
Let's compare `VGG7` and `VGG16` performance on `MNIST` dataset with different layouts of convolutional blocks. For each combination of layout and model class we train model for 1000 iterations and repeat that process 10 times. 

Firtsly, import classes from `dataset` to create pipelines:

.. code-block:: python

    from dataset import B, C, V, F, Config
    from dataset.opensets import MNIST
    from dataset.models.tf import VGG7, VGG16

Define model config. All parameters that we want to vary we define as ``C('parameter_name')``: 

.. code-block:: python

    model_config = {
        'inputs/images': {
            'shape': (28, 28, 1),
            'type': 'float32',
            'name': 'reshaped_images'
        },
        'inputs/labels': {
            'classes': 10,
            'type': 'int32',
            'transform': 'ohe',
            'name': 'targets'
        },
        'input_block/inputs': 'images',
        'output/ops': ['accuracy'],
        'loss': 'ce',
        'optimizer': 'Adam',
        'model_config/body/block/layout': C('layout')
    }

Strictly saying, the whole ``model_config`` with different ``'model_config/body/block/layout'`` is a pipeline parameter but due to a substitution rule of named expressions you can define named expression inside of `dict` or `Config` that is used as action parameter (See :doc:`Named expressions <../intro/named_expr>`).

Define dataset and train pipeline:

.. code-block:: python

    mnist = MNIST()

    feed_dict = {'images': B('images'),
                 'labels': B('labels')}

    vgg7_train = (mnist.train.p
                  .init_variable('loss', init_on_each_run=list)
                  .init_model('dynamic', C('model_class'), 'model', model_config)
                  .train_model('model', feed_dict=feed_dict, fetches='loss', save_to=V('loss'), mode='a')
                  .run(batch_size=32, shuffle=True, n_epochs=None, lazy=True)
                 )

Action parameters that we want to vary we define as ``C('model_class')``. Note that to specify parameters of batch generating
``run`` action must be defined with ``lazy=True``.

Create instance of `Research` class and add train pipeline:

.. code-block:: python

    research = Research()
    research.add_pipeline(vgg7_train, variables='loss', name='train')

We define parameter ``variables`` as ``'loss'`` to save pipeline variable with that name after training with each parameter configuration.
Parameter ``variables`` defines pipeline name inside ``research``.

All parameter combinations we define through the dict where key is a parameter name and value is list of possible parameter values.
Create grid of parameters and add to ``research``: 

.. code-block:: python

    grid_config = {'model_class': [VGG7, VGG16], 'layout': ['cna', 'can']}
    research.add_grid_config(grid_config)

In order to control test accuracy we create test pipeline and add it to ``research``:

.. code-block:: python

    vgg7_test = (mnist.test.p
             .init_variable('accuracy', init_on_each_run=list)
             .import_model('model', C('import_model_from'))
             .predict_model('model', feed_dict=feed_dict, fetches='output_accuracy', save_to=V('accuracy'), mode='a')
             .run(batch_size=100, shuffle=True, n_epochs=None, lazy=True)
            )

    research.add_pipeline(vgg7_test, variables='accuracy', name='test', import_model_from='train')

Note that we use ``C('import_model_from')`` in ``import_model`` action and add test pipeline with parameter ``import_model_from='train'``.
All ``kwargs`` in ``add_pipeline`` are used to define parameters that depends on other pipeline in the same way.

Method ``run`` starts computations:

.. code-block:: python

    research.run(n_reps=10, n_iters=1000, name='my_research'))

All result will be saved into ``my_research`` folder.

Parallel runnings
-----------------

Method ``run`` of ``Research`` has some additional parameters to allow run pipelines with different configs in parallel.
The first one is ``n_workers``. If you want to run pipelines in two different processes, run the following command:

.. code-block:: python

    research.run(n_reps=10, n_iters=1000, n_workers=2, name='my_research'))

API
---

See :doc:`Research API <../api/dataset.research>`.
