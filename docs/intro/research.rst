===========
Research
===========

Research class is intended for multiple running of the same pipelines with different parameters in order to get some metrics value.

Basic usage
-----------
Let's compare `VGG7` and `VGG16` performance on `MNIST` dataset with different layouts of convolutional blocks. For each combination of layout and model class, we train model for 1000 iterations and repeat that process 10 times. 

Firstly, import classes from `dataset` to create pipelines:

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
                  .train_model('model', feed_dict=feed_dict, fetches='loss', save_to=V('loss'), mode='w')
                  .run(batch_size=32, shuffle=True, n_epochs=None, lazy=True)
                 )

Action parameters that we want to vary we define as ``C('model_class')``. Note that to specify parameters of batch generating
``run`` action must be defined with ``lazy=True``.

Create an instance of `Research` class and add train pipeline:

.. code-block:: python

    research = Research()
    research.add_pipeline(vgg7_train, variables='loss', name='train')

Parameter ``name`` defines pipeline name inside ``research``. At each iteration that pipeline will be executed with ``.next_batch()`` and all ``variables`` from the pipeline will be saved so that variables must be added with ``mode='w'``.

All parameter combinations we define through the dict where a key is a parameter name and value is a list of possible parameter values.
Create a grid of parameters and add to ``research``: 

.. code-block:: python

    grid_config = {'model_class': [VGG7, VGG16], 'layout': ['cna', 'can']}
    research.add_grid_config(grid_config)

In order to control test accuracy we create test pipeline and add it to ``research``:

.. code-block:: python

    vgg7_test = (mnist.test.p
             .init_variable('accuracy', init_on_each_run=list)
             .import_model('model', C('import_model_from'))
             .predict_model('model', feed_dict=feed_dict, fetches='output_accuracy', save_to=V('accuracy'), mode='a')
             .run(batch_size=100, shuffle=True, n_epochs=1, lazy=True)
            )

    research.add_pipeline(vgg7_test, variables='accuracy', name='test', run=True, exec_for=100, import_model_from='train')

That pipeline will be executed with ``.run()`` at each 100 iterations because of parameters ``run=True``  and ``exec_for=100``. Pipeline variable ``accuracy`` will be saved after each execution. In order to add a mean value of accuracy on test dataset, you can define a function

.. code-block:: python

    def accuracy(pipeline):
        import numpy as np
        acc = pipeline.get_variable('accuracy')
        return {'mean_accuracy': np.mean(acc)}

and then add test pipeline as

.. code-block:: python

    research.add_pipeline(vgg7_test, variables='accuracy', name='test', run=True, exec_for=100, post_run=accuracy, import_model_from='train')

``post_run`` function must get pipeline as a parameter and return dict. That function will be executed for pipeline after each run and result will be saved.


Note that we use ``C('import_model_from')`` in ``import_model`` action and add test pipeline with parameter ``import_model_from='train'``.
All ``kwargs`` in ``add_pipeline`` are used to define parameters that depend on another pipeline in the same way.

Method ``run`` starts computations:

.. code-block:: python

    research.run(n_reps=10, n_iters=1000, name='my_research', progress_bar=True))

All results will be saved as ``my_research/{config_alias}/{number_of_repetition}/{pipeline_name}_final`` as dict where keys are variable names and values are lists of corresponding values. 

Parallel runnings
-----------------

Method ``run`` of ``Research`` has some additional parameters to allow run pipelines with different configs in parallel.
The first one is ``n_workers``. If you want to run pipelines in two different processes, run the following command:

.. code-block:: python

    research.run(n_reps=10, n_iters=1000, n_workers=2, name='my_research'))

Moreover, you can specify workers and define as ``n_workers`` as a list of dicts or Configs. Each worker will add the corresponding element of the list to pipeline config:

.. code-block:: python

    n_workers = [Config(model_config=dict(session=dict(config=tf.ConfigProto(gpu_options=tf.GPUOptions(visible_device_list=str(i)))))) for i in range(2)]
    research.run(n_reps=10, n_iters=1000, n_workers=n_workers, name='my_research'))

In that case, two workers will run pipelines in different processes on different GPU.

Another way of parallel running
--------------------------------

If you have a heavy preprocessing you can use one prepared batch for few pipelines with different configs. In that case, you must define ``root_pipeline`` that contains common actions without variable parameters:

.. code-block:: python

    train_root = mnist.train.p.run(BATCH_SIZE, shuffle=True, n_epochs=1, lazy=True) 

and ``branch_pipeline`` that will use prepared batch from ``root_pipeline`` and can contain variable parameters:

.. code-block:: python

    train_branch = (Pipeline()
            .init_variable('loss', init_on_each_run=list)
            .init_variable('accuracy', init_on_each_run=list)
            .init_model('dynamic', ResNet18, 'conv', config=model_config)
            .train_model('conv', 
                         fetches=['loss', 'output_accuracy'], 
                         feed_dict={'images': B('images'), 'labels': B('labels')},
                         save_to=[V('loss'), V('accuracy')], mode='w')

    research.add_pipeline(train_root, train_branch, variables=['loss', 'accuracy'], name='train')
)

In order to specify number of branches define ``n_branches`` parameter:
    
.. code-block:: python

    mr.run(n_reps=1, n_iters=1000, n_branches=2, name='branches', progress_bar=True)

As ``n_workers`` parameter you can define ``n_branches`` as a list of dicts or Configs that will be appended to corresponding branches.

API
---

See :doc:`Research API <../api/dataset.research>`.
