===========
Research
===========

Research class allows you to easily:
* experiment with models parameters combinations
* run multiple pipeline configurations (e.g. simultanious train and test workflow)
* add functions, customizing research process
* save and load results of experiments in a unified form

Basic usage
-----------
Let's compare `VGG7` and `VGG16` performance on `MNIST` dataset with
different layouts of convolutional blocks. For each combination of
layout and model class, we train model for 1000 iterations, repeat
that process 10 times and save accuracy and loss on train and accuracy on test.

Firstly, import classes from `batchflow` to create pipelines:

.. code-block:: python

    from batchflow import B, C, D, V, F, Config
    from batchflow.opensets import MNIST
    from batchflow.models.tf import VGG7, VGG16

    from batchflow.research import Research
    from batchflow.research import ResearchPipeline as RP
    from batchflow.research import ResearchIteration as RI

Define model config. All parameters that we want to vary we define
as ``C('parameter_name')``:

.. code-block:: python

model_config={
    'inputs/images/shape': B('image_shape'),
    'inputs/labels/classes': D('num_classes'),
    'inputs/labels/name': 'targets',
    'initial_block/inputs': 'images',
    'body/block/layout': C('layout')
}

Strictly saying, the whole ``model_config`` with different
``'body/block/layout'`` is a pipeline parameter but due
to a substitution rule of named expressions you can define
named expression inside of `dict` or `Config` that is used as action parameter
(See :doc:`Named expressions <../intro/named_expr>`).

Define a dataset and train a pipeline:

.. code-block:: python

    mnist = MNIST()

    train_ppl = (mnist.train.p
                 .init_variable('loss', default=[])
                 .init_model('dynamic', C('model'), 'conv', config=model_config)
                 .to_array()
                 .train_model('conv', images=B('images'), labels=B('labels'),
                              fetches='loss', save_to=V('loss', mode='w'))
                 .run_later(64, shuffle=True, n_epochs=None)
                )

Action parameters that we want to vary we define as ``C('parameter_name')``. Note
that to specify parameters of batch generating ``run_later`` action must be defined.

Create an instance of `Research` class and add train pipeline:

.. code-block:: python

    research = Research()
    research.add_pipeline(train_ppl, variables='loss', name='train')

Parameter ``name`` defines pipeline name inside ``research``. At each iteration
that pipeline will be executed with ``.next_batch()`` and all ``variables`` from the pipeline
will be saved so that variables must be added with ``mode='w'``.

All parameter combinations we define through the dict where a key is
a parameter name and value is a list of possible parameter values.
Create a grid of parameters and add to ``research``:

.. code-block:: python

    domain = Domain({'model_class': [VGG7, VGG16], 'layout': ['cna', 'can']})
    research.add_domain(domain, n_reps=10)

To create more complex domains that can be updated in the process of research
you can use ``Option`` class and ``update_method`` of ``Research``.

Each experiment with the same config will be repeated 10 times because of `n_reps` parameter.

You can get all variants of config by ``list(domain.iterator)``:

::

    [ConfigAlias({'layout': 'cna', 'repetition': '0', 'model': 'VGG7'}),
     ConfigAlias({'layout': 'cna', 'repetition': '0', 'model': 'VGG16'}),
     ConfigAlias({'layout': 'can', 'repetition': '0', 'model': 'VGG7'}),
     ConfigAlias({'layout': 'can', 'repetition': '0', 'model': 'VGG16'})]

Each element is a ConfigAlias. It's a Config dict of parameter values
and dict with aliases for parameter values.

In order to control test accuracy we create test pipeline and add it
to ``research``:

.. code-block:: python

    test_ppl = (mnist.test.p
                .init_variable('predictions')
                .init_variable('metrics')
                .import_model('conv', C('import_from'))
                .to_array()
                .predict_model('conv', 
                               images=B('images'), labels=B('labels'),
                               fetches='predictions', save_to=V('predictions'))
                .gather_metrics('class', targets=B('labels'), predictions=V('predictions'), 
                               fmt='logits', axis=-1, save_to=V('metrics'))
                .run_later(64, shuffle=True, n_epochs=1))

    research.add_pipeline(test_ppl, variables='metrics', name='test', run=True,
                          execute=100, import_model=RP('train'))

That pipeline will be executed with ``.run()`` each 100 iterations because
of parameters ``run=True``  and ``execute=100``. Pipeline variable ``metrics``
will be saved after each execution. In order to add a mean value of accuracy
on the whole test dataset, you can add ``get_metrics`` method into research:

.. code-block:: python

    research.get_metrics(pipeline='test', metrics_var='metrics', metrics_name='accuracy',
                         returns='accuracy', execute=100)

You also can add into pipeline your custom functions and specify ``args`` and ``kwargs`` through
`ResearchNamedExpression`-child classes.

Note that we use ``C('import_model')`` in ``import_model`` action
and add test pipeline with parameter ``import_model=RP('train')``.
All ``kwargs`` in ``add_pipeline`` will be substituted to pipeline
configs so we can use named expression ``RP('train')`` that will be
evaluated and transformed to actual pipeline object.

Method ``run`` starts computations:

.. code-block:: python

    research.run(n_iters=1000, name='my_research', bar=True)

All results will be saved as
``{research_name}/results/{config_alias}/{unitname}_{iteration}``
as pickled dict (by dill) where keys are variable names and values are lists
of corresponding values.

There is method ``load_results`` to create ``pandas.DataFrame`` with results
of the research.

Parallel runnings
-----------------

If you have a lot of GPU devices (say, 4) you can do research faster,
just define in ``run`` method ``workers=4``, ``devices = [0, 1, 2, 3]``
as a list of available devices and add ``device=C('device')`` into model
config. In that case you can run 4 jobs in parallel!

.. code-block:: python

    research.run(n_iters=1000, workers=4, devices=[0,1,2,3], name='my_research', bar=True)

In that case, two workers will execute tasks in different processes
on different GPU.

Another way of parallel running
--------------------------------

If you have heavy loading you can do it just one time for few pipelines
with models. In that case devide pipelines into root and branch:

.. code-block:: python

    mnist = MNIST()
    train_root = mnist.train.p.run_later(64, shuffle=True, n_epochs=None)

    train_ppl = (Pipeline()
                 .init_variable('loss', default=[])
                 .init_model('dynamic', C('model'), 'conv', config=model_config)
                 .to_array()
                 .train_model('conv', images=B('images'), labels=B('labels'),
                              fetches='loss', save_to=V('loss', mode='w'))
                 .run_later(64, shuffle=True, n_epochs=None)
                )


Then define research in the following way:

.. code-block:: python

    research = (Research()
        .add_pipeline(root=train_root, branch=train_branch, variables='loss', name='train')
        .add_pipeline(test_ppl, variables='metrics', name='test', run=True, execute=100, import_model=RP('train'))
        .add_domain(domain, n_reps=2)
        .get_metrics(pipeline='test', metrics_var='metrics', metrics_name='accuracy',
                     returns='accuracy', execute=100)
    )

And now you can define the number of branches in each worker:

.. code-block:: python

    research.run(n_iters=1000, workers=2, branches=2, devices=[0,1,2,3], name='my_research', bar=True)


Dumping of results and logging
--------------------------------

By default if unit has varaibles or returns then results
will be dumped at last iteration. But there is unit parameter dump
that allows to save result not only in the end. It defines as execute
parameter. For example, dump train results each 200 iterations.
Besides, each research has log file. In order to add information about
unit execution and dumping into log, define ``logging=True``.

.. code-block:: python

    research = (Research()
        .add_pipeline(root=train_root, branch=train_template,
                      variables='loss', name='train', dump=200)
        .add_pipeline(test_ppl,
                      variables='accuracy', name='test', run=True, execute=100, import_from=RP('train'), logging=True)
        .add_domain(domain, n_reps=2)
        .get_metrics(pipeline='test', metrics_var='metrics', metrics_name='accuracy',
                     returns='accuracy', execute=100)
    )

    research.run(n_iters=1000, workers=2, branches=2, devices=[0,1,2,3], name='my_research', bar=True)

First worker will execute two branches on GPU 0 and 1
and the second on the 2 and 3.

Functions on root
--------------------------------

All functions and pipelines if branches > 0 executed in parallel
threads so sometime it can be a problem. In order to allow run
function in main thread there exists parameter on_root. Function
that will be added with on_root=True will get iteration, experiments
and kwargs. experiments is a list of experiments that was defined above
(OrderedDict of ExecutableUnits). Simple example of usage:

.. code-block:: python

    def on_root(iteration):
        print("On root", iteration)

    research = (Research()
        .add_function(on_root, on_root=True, execute=10, iteration=RI(), logging=True)
        .add_pipeline(root=train_root, branch=train_template,
                      variables='loss', name='train', dump=200)
        .add_pipeline(test_ppl,
                      variables='accuracy', name='test', run=True, execute=100, import_from=RP('train'), logging=True)
        .add_domain(domain)
        .get_metrics(pipeline='test', metrics_var='metrics', metrics_name='accuracy',
                     returns='accuracy', execute=100)
    )

That function will be executed just one time on 10 iteration
and will be executed one time for all branches in task.

.. code-block:: python

    research.run(n_iters=100, workers=2, branches=2, devices=[0,1,2,3], name='my_research', bar=True)

Logfile:

::

    INFO     [2018-05-15 14:18:32,496] Distributor [id:5176] is preparing workers
    INFO     [2018-05-15 14:18:32,497] Create queue of jobs
    INFO     [2018-05-15 14:18:32,511] Run 2 workers
    INFO     [2018-05-15 14:18:32,608] Start Worker 0 [id:26021] (devices: [0, 1])
    INFO     [2018-05-15 14:18:32,709] Start Worker 1 [id:26022] (devices: [2, 3])
    INFO     [2018-05-15 14:18:41,722] Worker 0 is creating process for Job 0
    INFO     [2018-05-15 14:18:49,254] Worker 1 is creating process for Job 1
    INFO     [2018-05-15 14:18:53,101] Job 0 was started in subprocess [id:26082] by Worker 0
    INFO     [2018-05-15 14:18:53,118] Job 0 has the following configs:
    {'layout': 'cna', 'model': 'VGG7'}
    {'layout': 'cna', 'model': 'VGG16'}
    INFO     [2018-05-15 14:18:59,267] Job 1 was started in subprocess [id:26130] by Worker 1
    INFO     [2018-05-15 14:18:59,281] Job 1 has the following configs:
    {'layout': 'can', 'model': 'VGG7'}
    {'layout': 'can', 'model': 'VGG16'}
    INFO     [2018-05-15 14:19:02,415] J 0 [26082] I 11: on root 'unit_0' [0]
    INFO     [2018-05-15 14:19:02,415] J 0 [26082] I 11: on root 'unit_0' [1]
    INFO     [2018-05-15 14:19:07,803] J 0 [26082] I 100: dump 'unit_0' [0]
    INFO     [2018-05-15 14:19:07,803] J 0 [26082] I 100: dump 'unit_0' [1]
    INFO     [2018-05-15 14:19:08,761] J 1 [26130] I 11: on root 'unit_0' [0]
    INFO     [2018-05-15 14:19:08,761] J 1 [26130] I 11: on root 'unit_0' [1]
    INFO     [2018-05-15 14:19:12,050] J 0 [26082] I 100: execute 'test' [0]
    INFO     [2018-05-15 14:19:12,051] J 0 [26082] I 100: execute 'test' [1]
    INFO     [2018-05-15 14:19:12,051] J 0 [26082] I 100: dump 'test' [0]
    INFO     [2018-05-15 14:19:12,051] J 0 [26082] I 100: dump 'test' [1]
    INFO     [2018-05-15 14:19:12,056] Job 0 [26082] was finished by Worker 0
    INFO     [2018-05-15 14:19:14,149] J 1 [26130] I 100: dump 'unit_0' [0]
    INFO     [2018-05-15 14:19:14,149] J 1 [26130] I 100: dump 'unit_0' [1]
    INFO     [2018-05-15 14:19:18,819] J 1 [26130] I 100: execute 'test' [0]
    INFO     [2018-05-15 14:19:18,819] J 1 [26130] I 100: execute 'test' [1]
    INFO     [2018-05-15 14:19:18,820] J 1 [26130] I 100: dump 'test' [0]
    INFO     [2018-05-15 14:19:18,820] J 1 [26130] I 100: dump 'test' [1]
    INFO     [2018-05-15 14:19:18,825] Job 1 [26130] was finished by Worker 1
    INFO     [2018-05-15 14:19:18,837] All workers have finished the work

API
---

See :doc:`Research API <../api/batchflow.research>`.
