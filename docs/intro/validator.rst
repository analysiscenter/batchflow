=========
Validator
=========

Getting started
===============
Validator is an instrument to unify the process of model training and validation.
There are two main reasons to wrap your model using classes from `Validator` submodule:

- make the structure of the model lifecycle clear,
- provide API to automized model training and validation.

All you need is to define class inherited from `Validator` and then you can use it in your scipts: ::

    validator = LithologyModel(dpcm=20, crop_length=10)
    train_dataset = validator.load_train_dataset(path='/train')
    validator.train(train_dataset, n_epochs=300)
    test_dataset = validator.load_test_dataset('/test')
    targets, predictions = validator.inference(test_dataset)
    validator.compute_metrics(targets, predictions, 'f1_score')

Also you can configure it by `yaml` file and execute the following code: ::

        from validator import MyValidator

        val = MyValidator('validator.yaml')
        val.run()
        print(val.metrics)

Basic example
=============

model_api.py
------------
::

    import ...

    class PascalValidator(Validator):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.model_config = ...
            self.dataset = PascalSegmentation(bar=True)

        def train_loader(self, **kwargs):
            return self.dataset.train

        def test_loader(self, **kwargs):
            return self.dataset.test

        def train(self, dataset):
            train_ppl = ... << dataset
            train_ppl.run()
            self.model = train_ppl.get_model_by_name('model')

        def inference(self, dataset):
            model = self.model
            test_ppl = ... << dataset
            test_ppl.run()
            targets = np.array(test_ppl.v('targets'))
            predictions = np.array(test_ppl.v('predictions'))
            return targets, predictions


validator.yaml
--------------
::

    train:
    test:
        metrics: accuracy


`val.metrics`
-------------

::

    {'accuracy': 0.0092724609375}

Validator class
===============

`Validator` child-class can implement the following methods:

`train_loader(self, path=None, **kwargs)`
-----------------------------------------

`path` and `kwargs` are from config `<train>`. In the example above we have empty value for `train` key, therefore `path=None` and `kwargs={}`. Let's define config:

::

    train:
        dataset: /path/to/dataset


In that case `path='/path/to/dataset'`. You also can define multiple parameters of `train_loader`: ::

    train:
        dataset:
        path: /path/to/dataset
        format: 'png'

Now `path='/path/to/dataset', kwargs={format: 'png'}`.
The output of the function will be used as `train_dataset` argument of `train` method. By default, it returns `path`.

`test_loader(self, path=None, **kwargs)`
----------------------------------------

The same as `train_loader` but for `test`.

`load_model(self, path=None, **kwargs)`
---------------------------------------

Loader for pretrained model. Let's make example above more complex::

    train:
    pretrained:
        path: /path/to/model
        device: cuda:0
    test:
        metrics: accuracy

In that case `path='/path/to/model'` and `kwargs={device: 'cuda:0'}`. The output of the function will be used as `train_output` argument of `inference` method. By default, it returns `path`. Note that when you define `pretrained` key in your config, train section will be skipped.

`train(self, train_dataset, **kwargs)`
--------------------------------------

Function that must contain the whole training process. Argument `train_dataset` is an output of `train_loader` method, dict `kwargs` is from config and doesn't include popped `dataset` key. Example::

    train:
        dataset: /path/to/data
        model: UNet
    test:
        metrics: accuracy

In that case `kwargs={model: 'UNet'}`. Method is executed when `pretrained` is not defined.

`inference(self, test_dataset, **kwargs)`
-------------------------------------------------------

Function that must contain the whole inference process. Argument `test_dataset` is an output of `test_loader` method. `kwargs` is from config and doesn't include popped `dataset` key.
Function returns `predictions` and `targets` in format that can be used with Batchflow metrics (see :doc:`metrics API <../api/batchflow.models.metrics>`).

Custom metrics
--------------

If you need to realize your custom metrics, add method like::

    def my_accuracy(self, target, prediction):
        return (target == prediction.argmax(axis=1)).mean()


To specify what metrics will be computed, add them into config::

    ...
      test:
        metrics:
            - accuracy
            - f1_score
            - my_accuracy
      accuracy:
        class: classification # BatchFLow class of metrics
        axis: 1               # Init parameters
      f1_score:
        class: classification
        axis: 1
        evaluate:            # Evaluate parameters
            agg: mean
            multiclass:


validator.yaml
==============

Generally has the following structure::


    train: (optional)
        dataset:
            - <dataset_param_0>: <value_0>
            ...
    pretrained: (optional)
        path: <model_path>
    test:
        dataset:
            - <dataset_param_0>: <value_0>
            ...
        metrics:
            - <metric_0>         # BatchFLow class of metrics because `metric_0` is also key of the first level of config
            - ...
            - <custom_metric_0>  # custom metric defined in Validator-child class
            ...
    metric_0
        class: <classification|segmentation|mask|instance|regression>
            <kwarg_0>: <value_0>
            <kwarg_1>: <value_1>
            ...
            evaluate:
                <metric_kwarg_0>: <value_0>
                <metric_kwarg_1>: <value_1>
    ...

Also you can define metrics in the following way: ::

    metrics: <metric_0>, <metric_1>, <custom_metric_0>, ...

Style guide
===========

To make your interfaces clearer, we propose one rule: use each of 5 methods to divide your model lify-cycle into clear blocks.
For example, there are several options to define data loading: `__init__`, `train`/`inference` but it's better when you use special methods 'train_loader'/'test_loader'.

To check that interface has necessary methods, you can call `check_api` method.
For example, call class method::

    MyValidator.check_api(methods=['train_loader', 'train'])

to check if methods `train_loader` and `train` are implemented in MyValidator class. By default, ::

    methods=['train', 'inference']

and warning will be issued if one of methods is not implemented. To raise exception instead of warning, use `warning=False`.

You also can check keys in validator config by `check_config` method: ::

    val = MyValidator('validator.yaml')
    val.check_config(keys=['train|pretrained', 'load_model])

Successful check means that class implements `load_model` method and one of 'train` and `pretrained`.