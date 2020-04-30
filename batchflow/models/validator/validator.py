""" API for models and validator """

from abc import ABCMeta, abstractmethod
import warnings

import yaml

from ..metrics import ClassificationMetrics, SegmentationMetricsByPixels, \
                      SegmentationMetricsByInstances, RegressionMetrics

METRICS = dict(
    classification=ClassificationMetrics,
    segmentation=SegmentationMetricsByPixels,
    mask=SegmentationMetricsByPixels,
    instance=SegmentationMetricsByInstances,
    regression=RegressionMetrics
)

def key_value(d):
    return list(d.keys())[0], list(d.values())[0]

class ModelAPI(metaclass=ABCMeta):
    """ Model API: train, inference and metrics

    **Attributes**

    config : dict
        model config from yaml file (see :class:`~.Validator`)
    """
    def __init__(self, config, name, task):
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

        self.custom_metric_values = {}
        self.metric_values = {}

    @abstractmethod
    def train(self, *args, **kwargs):
        pass

    @abstractmethod
    def inference(self, *args, **kwargs):
        pass

    def run(self):
        """ Run validator """
        self.init()

        if self.pretrained is None:
            self.train(self.train_dataset, self.model_path)

        if self.pretrained:
            model_path = self.pretrained['model']
        else:
            model_path = self.model_path

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
    """ Validator class

    **Attributes**

    config_path : str
        path yaml file with config. Has the following structure:
        ```
        - <task_name_1>
            - <model_name_1>
                class: <class>
                train: (optional)
                    dataset: <dataset_path>
                    model: <model_path>
                pretrained: (optional)
                    model: <model_path>
                validate:
                    dataset: <dataset_path>
                    metrics:
                        - <classification|segmentation|mask|instance|regression>
                            <classification_kwarg_1>: <value_1>
                            <classification_kwarg_2>: <value_2>
                            ...
                            evaluate:
                                - <metric_name_1>:
                                    <metric_kwarg_1>: <value_1>
                                    <metric_kwarg_2>: <value_2>
                                - <metric_name_2>:
                                    ...
                                ...
                    custom_metrics: (optional)
                        - <custom_metric_1>
                        - <custom_metric_2>
                        ...
            - <model_name_2>
                ...
        - <task_name_2>
            ...
        ```

    **How to get metrics values**

        After execution Validator will have attribute `results` with the resulting ModelAPI instances.
        To get metrics, call `.metrics` or `.custom_metrics` method.
    """
    def __init__(self, config_path='validator.yaml'):
        self.config_path = config_path
        with open(self.config_path) as file:
            self.config = yaml.load(file, Loader=yaml.Loader)
        self.results = {}

    def start(self):
        """ Start validator """
        import model_api #pylint: disable=unused-import
        for task in self.config:
            task_name, task_config = key_value(task)
            self.results[task_name] = {}
            for model in task_config:
                model_name, config = key_value(model)
                validator = eval('model_api.'+config['class'])(config, model_name, task_name) #pylint: disable=eval-used
                self.results[task_name][model_name] = validator
                validator.run()

    def _metrics(self, custom_metrics=False):
        attr = 'custom_metric_values' if custom_metrics else 'metric_values'
        return {
            task_name:
            {model_name: getattr(self.results[task_name][model_name], attr) for model_name in self.results[task_name]}
            for task_name in self.results
        }

    @property
    def metrics(self):
        """ Metrics.

        Returns
        -------
        metrics: dict
            metrics dict. Has the following structure:
            ```
            task_name_0:
                model_name_0:
                    metric_name_0: value_0
                    metric_name_1: value_1
                    ...
                model_name_1:
                    ...
                ...
            task_name_1:
                ...
            ```
        """
        return self._metrics(custom_metrics=False)

    @property
    def custom_metrics(self):
        """ Metrics.

        Returns
        -------
        metrics: dict
            custom metrics dict. Has the following structure:
            ```
            task_name_0:
                model_name_0:
                    metric_name_0: value_0
                    metric_name_1: value_1
                    ...
                model_name_1:
                    ...
                ...
            task_name_1:
                ...
            ```
        """
        return self._metrics(custom_metrics=True)
