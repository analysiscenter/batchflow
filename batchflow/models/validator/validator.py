import os
import sys
import yaml
import numpy as np
import yaml
import warnings
from abc import ABCMeta, abstractmethod

from ..metrics import ClassificationMetrics, SegmentationMetricsByPixels, SegmentationMetricsByInstances, RegressionMetrics

METRICS = dict(
    classification=ClassificationMetrics,
    segmentation=SegmentationMetricsByPixels,
    mask=SegmentationMetricsByPixels,
    instance=SegmentationMetricsByInstances,
    regression=RegressionMetrics
)

def key_value(d):
    return list(d.keys())[0], list(d.values())[0]

class AbstractValidator(metaclass=ABCMeta):
    @abstractmethod
    def train(*args, **kwargs):
        pass

    @abstractmethod
    def inference(*args, **kwargs):
        pass
    
class BaseValidator(AbstractValidator):
    def __init__(self, validator_config, validator_class, task_name):
        self.validator_config = validator_config
        self.validator_class = validator_class
        self.task_name = task_name
        self.model_path = None

        if 'train' in validator_config and 'pretrained' in validator_config:
            warnings.warn("Both 'train' and 'pretrained' was founded \
            for {} in {} so train stage will be skipped".format(validator_class, task_name))
        
        if 'train' in validator_config:
            self.train_dataset = self.validator_config['train']['dataset']
            self.model_path = self.validator_config['train'].get('model')

        self.pretrained = validator_config.get('pretrained')
        self.validate_dataset = self.validator_config['validate']['dataset']
        self.metrics = self.validator_config['validate']['metrics']
        self.custom_metrics = self.validator_config['validate'].get('custom_metrics')
        
    def run(self):
        print("Task: {}. Validator: {}".format(self.task_name, self.validator_class))
        print("Call init...")
        self.init()

        if self.pretrained is None:
            print('Start train...')
            self.train(self.train_dataset, self.model_path)
        
        if self.pretrained:
            model_path = self.pretrained['model']
            pretarined_flag = ' (pretrained)'
        else:
            model_path = self.model_path
            pretarined_flag = ''
        
        print('Start inference on validation dataset...')
        self.inference(self.validate_dataset, model_path)
        
        self.metric_values = {}
        for _metric in self.metrics:
            metric_class, params = key_value(_metric)
            evaluate = params.pop('evaluate')
            metrics = METRICS[metric_class](self.targets, self.predictions, **params)
            self.metric_values[metric_class] = {}
            for item in evaluate:
                if isinstance(item, dict):
                    metric_name = list(item.keys())[0]
                    kwargs = list(item.values())[0]
                else:
                    metric_name = item
                    kwargs = dict()
                self.metric_values[metric_class][metric_name] = metrics.evaluate(metric_name, **kwargs)

        self.custom_metric_values = {}
        for _metric in self.custom_metrics or []:
            value = getattr(self, _metric)(self.targets, self.predictions)
            self.custom_metric_values[_metric] = value

    @classmethod
    def start(cls, config_path='validator.yaml'):
        import validator_api

        with open(config_path) as file:
            config = yaml.load(file, Loader=yaml.Loader)
        for task in config:
            task_name, task_config = key_value(task)
            for validator_item in task_config:
                validator_name, validator_config = key_value(validator_item)
                validator = eval('validator_api.'+validator_config['class'])(validator_config, validator_name, task_name)
                validator.run()
                print(validator.metric_values)
                print(validator.custom_metric_values)
            
                    