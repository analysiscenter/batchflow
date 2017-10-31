===================================
Welcome to Dataset's documentation!
===================================
`Dataset` helps you conveniently work with random or sequential batches of your data
and define data processing and machine learning workflows even for datasets that do not fit into memory.

Main features:

* flexible batch generaton
* deterministic and stochastic pipelines
* datasets and pipelines joins and merges
* data processing actions
* flexible model configuration
* within batch parallelism
* batch prefetching
* ready to use ML models and proven NN architectures
* convenient layers and helper functions to build custom models.


Contents
========
.. toctree::
   :maxdepth: 2
   :titlesonly:

   intro/intro
   intro/classes
   api/dataset


Basic usage
===========
.. code-block:: python

    my_workflow = my_dataset.pipeline()
                    .load('/some/path')
                    .do_something()
                    .do_something_else()
                    .some_additional_action()
                    .save('/to/other/path')

The trick here is that all the processing actions are lazy. They are not executed until their results are needed, e.g. when you request a preprocessed batch:

.. code-block:: python

    NUM_ITERS = 1000
    for i in range(NUM_ITERS):
        processed_batch = my_workflow.next_batch(BATCH_SIZE, shuffle=True, n_epochs=None)
        # only now the actions are fired and data is changed with the workflow defined earlier


Installation
============

.. note:: `Dataset` module is in the beta stage. Your suggestions and improvements are very welcome.

.. note:: `Dataset` supports python 3.5 or higher.


Git submodule
-------------
In many cases it is much more convenient to install `dataset` as a submodule in your project repository than as a system python package.

.. code-block:: shell

    git submodule add https://github.com/analysiscenter/dataset.git
    git submodule init
    git submodule update

After that you can import it as a python module:

.. code-block:: python

    import dataset as ds

If your python file is located in a subdirectory, you might need to add a path to `dataset`:

.. code-block:: python

    import sys
    sys.path.append("..")
    import dataset as ds
