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

class AbstractModelAPI(metaclass=ABCMeta):
    @abstractmethod
    def train(*args, **kwargs):
        pass

    @abstractmethod
    def inference(*args, **kwargs):
        pass
    
class ModelAPI(AbstractModelAPI):
    def __init__(self, config, name, task_name):
        self.config = config
        self.name = name
        self.task = task
        self.model_path = None

        if 'train' in config and 'pretrained' in config:
            warnings.warn("Both 'train' and 'pretrained' was founded \
            for {} in {} so train stage will be skipped".format(name, task))
        
        if 'train' in config:
            self.train_dataset = self.config['train']['dataset']
            self.model_path = self.config['train'].get('model')

        self.pretrained = config.get('pretrained')
        self.validate_dataset = self.config['validate']['dataset']
        self.metrics = self.config['validate']['metrics']
        self.custom_metrics = self.config['validate'].get('custom_metrics')
        
    def run(self):
        print("Task: {}. ModelAPI: {}".format(self.task, self.name))
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


class Validator:
    def __init__(self, config_path='validator.yaml'):
        self.config_path = config_path
        self.results = {}

    def start(self):
        import validator_api

        with open(self.config_path) as file:
            self.config = yaml.load(file, Loader=yaml.Loader)
        for task in self.config:
            task_name, task_config = key_value(task)
            self.results[task_name] = {}
            for model in task_config:
                model_name, config = key_value(model)
                validator = eval('validator_api.'+config['class'])(config, model_name, task_name)
                self.results[task_name][model_name] = validator
                validator.run()

    def _metrics(self, custom_metrics=False):
        attr = 'custom_metric_values' if custom_metrics else 'metric_values'
        return {
            task_name: 
            {model_name: getattr(self.results[task_name][model_name], attr) for model_name in self.results[task_name]}
            for task_name in self.results
        }

    def metrics(self):
        return self._metrics(custom_metrics=False)

    def custom_metrics(self):
        return self._metrics(custom_metrics=True)
                    