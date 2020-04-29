# Validator

Validator is an instrument to unify the process of model training and validation. To make model API, you need:
- make validator-class inherited from `BaseValidator` with `train` and `inference` methods (see details below),
- configure validator by `validator.yaml`.

## Validator class
How to cook your own validator:
- Define `train` method with required argument `dataset_path` and optional argument `model_path`. Here you should place the whole train process of the model. If you will save model, use `model_path`, otherwise define `self.from_train` with all necessary variables that you can use at inference stage.
- Define `inference` method with required arguments `dataset_path` and optional argument `model_path`. `model_path` will be the path to pretrained model (if `pretrained` is defined in  `validator.yaml`) or the path which is the same as `model_path` in `train` method. In `inference` method you must define `self.predictions` and `self.targets` that will be numpy arrays.
- Define your own metrics with 

## `validator.yaml`
Validator is used to determine the whole procedure of validation.
Let’s consider an example
```yaml
- Pascal Segmentation: # the task name (you can have different tasks with different models in your repo
       - PascalValidator: # name of the validator
            class: PascalValidator # validator from validator_api.py
            pretrained: # if you will use pretrained model, train block is not necessary
              model: ./model.torch
            validate:
              dataset: /path/to/dataset
              metrics: # batchflow metrics to perform validation
                - classification: # metrics class. Below define parameters
                    axis: 1
                    fmt: 'logits'
                    num_classes: 22
                    evaluate:
                        - f1_score:
                            agg: mean
                            multiclass:
                        - accuracy
              custom_metrics: # your custom metrics
                - accuracy
        - PascalValidator2:
            class: PascalValidator
            train: # train parameters that will be substituted into train method
              dataset: /path/to/dataset
              model: ./model2.torch # that model will be used as argument in inference method
            validate:
              dataset: /path/to/dataset
              metrics:
                - classification:
                    axis: 1
                    fmt: 'logits'
                    num_classes: 22
                    evaluate:
                        - accuracy

```

## How to use
Define your validator class inherited from BaseValidator in `validator_api.py` and config in `validator.yaml`
Then from `validator` folder execute class method of BaseValidator
```
BaseValidator.start()
```
