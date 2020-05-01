# Validator

Validator is an instrument to unify the process of model training and validation.
There are two main reasons to wrap your model using classes from `Validator` submodule:
- make the structure of the model lifecycle clear,
- provide API to automized model training and validation.

All you need is:

* define `model_api.py` file with model inherited from `ModelAPI`,
* configure it by `validator.yaml` file in the same folder,
* execute the following code from the same folder:
```python
from batchflow.models.validator import Validator

val = Validator()
val.start()
print(val.metrics)
```

## Basic example:
**model_api.py**
```python
import ...

class PascalValidator(ModelAPI):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_config = ...
        self.ds = PascalSegmentation(bar=True)

    def train_loader(self, **kwargs):
        return self.ds.train

    def test_loader(self, **kwargs):
        return self.ds.test

    def train(self, ds):
        train_ppl = ... << ds
        train_ppl.run()
        return train_ppl.get_model_by_name('model')

    def inference(self, ds, model):
        test_ppl = ... << ds
        test_ppl.run()
        targets = np.array(test_ppl.v('targets'))
        predictions = np.array(test_ppl.v('predictions'))
        return targets, predictions
```

**validator.yaml**
```yaml
- Pascal Segmentation: 
  - PascalValidator:
      class: PascalValidator
      train:
      test:
        metrics:
        - classification:
            evaluate:
              - accuracy
```

**val.metrics**
```
{'Pascal Segmentation':
       {'PascalValidator':
              {'classification': {'accuracy': 0.0092724609375}
       }
}
```

## **model_api**
`ModelAPI` child-class can implement the following methods:
#### `train_loader(self, path=None, **kwargs)`
`path` and `kwargs` are from config `<task_name>/<model_name>/<train>`. In the example above we have empty value for `train` key, therefore `path=None` and `kwargs={}`. Let's define config:
```yaml
- Pascal Segmentation: 
  - PascalValidator:
      class: PascalValidator
      train:
        dataset: /path/to/dataset
```
In that case `path='/path/to/dataset'`. You also can define multiple parameters of `train_loader`:
```yaml
- Pascal Segmentation: 
  - PascalValidator:
      class: PascalValidator
      train:
         dataset:
           path: /path/to/dataset
           format: 'png'
```
Now `path='/path/to/dataset', kwargs={format: 'png'}`.
   
The output of the function will be used as `train_dataset` argument of `train` method. By default, it returns `path`.
       
#### `test_loader(self, path=None, **kwargs)`

The same as `train_loader` but for `test`.

#### `load_model(self, path=None, **kwargs)`

Loader for pretrained model. Let's make example above more complex:
```yaml
- Pascal Segmentation: 
  - PascalValidator:
      class: PascalValidator
      train:
      pretrained:
        path: /path/to/model
        device: cuda:0
      test:
        metrics:
        - classification:
            evaluate:
              - accuracy
```
In that case `path=/path/to/model` and `kwargs={device: cuda:0}`. The output of the function will be used as `train_output` argument of `inference` method. By default, it returns `path`. Note that when you define `pretrained` key in your config, train section will be skipped.

#### `train(self, train_dataset, **kwargs)`

Function that must contain the whole training process. Argument `train_dataset` is an output of `train_loader` method, dict `kwargs` is from config and doesn't include popped `dataset` key. Example:
```yaml
- Pascal Segmentation: 
  - PascalValidator:
      class: PascalValidator
      train:
        dataset: /path/to/data
        model: UNet
      test:
        metrics:
        - classification:
            evaluate:
              - accuracy
```
In that case `kwargs={model: 'UNet'}`. If `pretrained` is not defined in config, output of that function will be used as `train_output` argument of `inference`.

#### `inference(self, test_dataset, train_output, **kwargs)`

Function that must contain the whole inference process. Argument `test_dataset` is an output of `test_loader` method, `train_output` is an output of `load_model` method for configs with `pretrained` or of `train` method, otherwise. `kwargs` is from config and doesn't include popped `dataset` key.
Function returns `predictions` and `targets` in format that can be used with [Batchflow metrics](https://github.com/analysiscenter/batchflow/tree/master/batchflow/models/metrics).

#### **Custom metrics**

If you need to realize your custom metrics, add method like
```python
    def my_accuracy(self, target, prediction):
        return (target == prediction.argmax(axis=1)).mean()
```

To specify what metrics will be computed, add them into config:
```
...
      test:
        metrics:
        - classification:         # batchflow class of metrics
            axis: 1               # init parameter of metric class
            fmt: 'logits'         # ...
              evaluate:
                - f1_score:       # metric to evaluate
                    agg: mean     # parameter of evaluate
                - accuracy
        custom_metrics:
        - my_accuracy             # metric from your class
```

## **validator.yaml**

Generally has the following structure:

```yaml
 - <task_name_0>
     - <model_name_0>
         class: <class>
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
             metrics: (metics from batchflow)
                 - <classification|segmentation|mask|instance|regression>
                     <classification_kwarg_0>: <value_0>
                     <classification_kwarg_1>: <value_1>
                     ...
                     evaluate:
                         - <metric_name_0>:
                             <metric_kwarg_0>: <value_0>
                             <metric_kwarg_1>: <value_1>
                         - <metric_name_1>:
                             ...
                         ...
             custom_metrics: (optional, metrics defined in ModelAPI child-class)
                 - <custom_metric_0>
                 - <custom_metric_1>
                 ...
     - <model_name_1>
         ...
 - <task_name_1>
 ...
 ```
