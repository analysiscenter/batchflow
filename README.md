[![License](https://img.shields.io/github/license/analysiscenter/batchflow.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Python](https://img.shields.io/badge/python-3.6-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.7-orange.svg)](https://pytorch.org)
[![codecov](https://codecov.io/gh/analysiscenter/batchflow/branch/master/graph/badge.svg)](https://codecov.io/gh/analysiscenter/batchflow)
[![PyPI](https://badge.fury.io/py/batchflow.svg)](https://badge.fury.io/py/batchflow)
[![Status](https://github.com/analysiscenter/batchflow/workflows/status/badge.svg)](https://github.com/analysiscenter/batchflow/actions?query=workflow%3Astatus)


# BatchFlow

`BatchFlow` helps you conveniently work with random or sequential batches of your data
and define data processing and machine learning workflows even for datasets that do not fit into memory.

For more details see [the documentation and tutorials](https://analysiscenter.github.io/batchflow/).

Main features:
- flexible batch generaton
- deterministic and stochastic pipelines
- datasets and pipelines joins and merges
- data processing actions
- flexible model configuration
- within batch parallelism
- batch prefetching
- ready to use ML models and proven NN architectures
- convenient layers and helper functions to build custom models
- a powerful research engine with parallel model training and extended experiment logging.

## Basic usage

```python
my_workflow = my_dataset.pipeline()
              .load('/some/path')
              .do_something()
              .do_something_else()
              .some_additional_action()
              .save('/to/other/path')
```
The trick here is that all the processing actions are lazy. They are not executed until their results are needed, e.g. when you request a preprocessed batch:
```python
my_workflow.run(BATCH_SIZE, shuffle=True, n_epochs=5)
```
or
```python
for batch in my_workflow.gen_batch(BATCH_SIZE, shuffle=True, n_epochs=5):
    # only now the actions are fired and data is being changed with the workflow defined earlier
    # actions are executed one by one and here you get a fully processed batch
```
or
```python
NUM_ITERS = 1000
for i in range(NUM_ITERS):
    processed_batch = my_workflow.next_batch(BATCH_SIZE, shuffle=True, n_epochs=None)
    # only now the actions are fired and data is changed with the workflow defined earlier
    # actions are executed one by one and here you get a fully processed batch
```


## Train a neural network
`BatchFlow` includes ready-to-use proven architectures like VGG, Inception, ResNet and many others.
To apply them to your data just choose a model, specify the inputs (like the number of classes or images shape)
and call `train_model`. Of course, you can also choose a loss function, an optimizer and many other parameters, if you want.
```python
from batchflow.models.torch import ResNet34

my_workflow = my_dataset.pipeline()
              .init_model('model', ResNet34, config={'loss': 'ce', 'classes': 10})
              .load('/some/path')
              .some_transform()
              .another_transform()
              .train_model('ResNet34', inputs=B.images, targets=B.labels)
              .run(BATCH_SIZE, shuffle=True)
```

For more advanced cases and detailed API see [the documentation](https://analysiscenter.github.io/batchflow/).


## Installation

> `BatchFlow` module is in the beta stage. Your suggestions and improvements are very welcome.

> `BatchFlow` supports python 3.6 or higher.

### Stable python package

With [poetry](https://python-poetry.org/)
```
poetry add batchflow
```

With old-fashioned [pip](https://pip.pypa.io/en/stable/)
```
pip3 install batchflow
```

### Development version

With [poetry](https://python-poetry.org/)
```
poetry add --editable git+https://github.com/analysiscenter/batchflow
```

With old-fashioned [pip](https://pip.pypa.io/en/stable/)
```
pip install --editable git+https://github.com/analysiscenter/batchflow
```

### Extras
Some `batchflow` functions and classed require additional dependencies.
In order to use that functionality you might need to install `batchflow` with extras (e.g. `batchflow[nn]`):

- image - working with image datasets and plotting
- nn - for neural networks (includes torch, torchvision, ...)
- datasets - loading standard datasets (MNIST, CIFAR, ...)
- profile - performance profiling
- jupyter - utility functions for notebooks
- research - multiprocess research
- telegram - for monitoring pipelines via a telegram bot
- dev - batchflow development (pylint, pytest, ...)

You can install several extras at once, like `batchflow[image,nn,research]`.


## Projects based on BatchFlow
- [SeismiQB](https://github.com/gazprom-neft/seismiqb) - ML for seismic interpretation
- [SeismicPro](https://github.com/gazprom-neft/SeismicPro) - ML for seismic processing
- [PetroFlow](https://github.com/gazprom-neft/petroflow) - ML for well interpretation
- [PyDEns](https://github.com/analysiscenter/pydens) - DL Solver for ODE and PDE
- [RadIO](https://github.com/analysiscenter/radio) - ML for CT imaging
- [CardIO](https://github.com/analysiscenter/cardio) - ML for heart signals


## Citing BatchFlow
Please cite BatchFlow in your publications if it helps your research.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1041203.svg)](https://doi.org/10.5281/zenodo.1041203)

```
Roman Khudorozhkov et al. BatchFlow library for fast ML workflows. 2017. doi:10.5281/zenodo.1041203
```

```
@misc{roman_kh_2017_1041203,
  author       = {Khudorozhkov, Roman and others},
  title        = {BatchFlow library for fast ML workflows},
  year         = 2017,
  doi          = {10.5281/zenodo.1041203},
  url          = {https://doi.org/10.5281/zenodo.1041203}
}
```
