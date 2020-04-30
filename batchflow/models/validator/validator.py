""" API for models and validator """

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

class ModelAPI:
    """ Model API: train, inference and metrics

    **Attributes**

    config : dict
        model config from yaml file (see :class:`~.Validator`)
    """
    def __init__(self, config, name, task):
        self.config = config
        self.name = name
        self.task = task

        if 'train' in config and 'pretrained' in config:
            warnings.warn("Both 'train' and 'pretrained' was founded \
            for {} in {} so train stage will be skipped".format(name, task))

        self._pretrained = self.config.get('pretrained', {})

        self._train_ds = {}
        if 'train' in self.config and self.config['train'] is not None:
            self._train_ds = self.config['train'].get('dataset', {})
            if isinstance(self._train_ds, str):
                self._train_ds = {'path': self._train_ds}

        self._test_ds = {}
        if 'test' in self.config and self.config['test'] is not None:
            self._test_ds = self.config['test'].get('dataset', {})
            if isinstance(self._test_ds, str):
                self._test_ds = {'path': self._test_ds}
            self._metrics = self.config['test'].get('metrics', {})
            self._custom_metrics = self.config['test'].get('custom_metrics', [])

        self.custom_metric_values = {}
        self.metric_values = {}

    def init(self):
        pass

    def train_loader(self, *args, path=None, **kwargs):
        return path

    def test_loader(self, *args, path=None, **kwargs):
        return path

    def load_model(self, path):
        return path

    def train(self, *args, **kwargs):
        pass

    def inference(self, *args, **kwargs):
        pass

    def compute_metrics(self):
        self.metric_values = {}
        for _metric in self._metrics:
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

    def compute_custom_metrics(self):
        self.custom_metric_values = {}
        for _metric in self._custom_metrics:
            value = getattr(self, _metric)(self.targets, self.predictions)
            self.custom_metric_values[_metric] = value

    def run(self):
        """ Run validator """
        self.init()

        if 'train' in self.config:
            self.train_dataset = self.train_loader(**self._train_ds)
            self.from_train = self.train(self.train_dataset)
        elif 'pretrained' in self.config:
            self.from_train = self.load_model(**self._pretrained)

        if 'test' in self.config:
            self.test_dataset = self.test_loader(**self._test_ds)
            self.inference(self.test_dataset, self.from_train)
            self.compute_metrics()
            self.compute_custom_metrics()



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
