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

class ModelAPI:
    """ Model API: train, inference and metrics

    **Attributes**

    config_path : str
        path to yaml file with config of the following structure:
        ```
        train: (optional)
            dataset:
                - <dataset_param_0>: <value_0>
                ...
        pretrained: (optional)
            path: <model_path>
        test: (optional)
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
        ```
    name : str
        name of the current model (from yaml)
    task : str
        name of the current task (from yaml)
    """
    def __init__(self, config_path):
        with open(config_path) as file:
            self.config = yaml.load(file, Loader=yaml.Loader)

        if 'train' in self.config and 'pretrained' in self.config:
            warnings.warn("Both 'train' and 'pretrained' was founded so train stage will be skipped")

        self._pretrained = self.config.get('pretrained', {})
        self._train_ds = {}
        if 'train' in self.config and self.config['train'] is not None:
            self._train_ds = self.config['train'].pop('dataset', {})
            if isinstance(self._train_ds, str):
                self._train_ds = {'path': self._train_ds}

        self._test_ds = {}
        if 'test' in self.config and self.config['test'] is not None:
            self._test_ds = self.config['test'].pop('dataset', {})
            if isinstance(self._test_ds, str):
                self._test_ds = {'path': self._test_ds}
            self._metrics = self.config['test'].pop('metrics', {})

        self.train_dataset = None
        self.from_train = None
        self.test_dataset = None
        self.targets = None
        self.predictions = None

        self.metrics = {}

    def train_loader(self, path=None, **kwargs):
        """ Train dataset loader.

        Parameters
        ----------
        path : str or None
            path to train dataset, defined in `train/dataset` section of config.

        Return
        ------
        path : str or None
            If function is not defined in child class, return `path`. It will be
            used as first argument of `train`.
         """
        _ = kwargs
        return path

    def test_loader(self, path=None, **kwargs):
        """ Test dataset loader.

        Parameters
        ----------
        path : str or None
            path to test dataset, defined in `test/dataset` section of config.

        Return
        ------
        path : str or None
            If function is not defined in child class, return `path`. It will be
            used as the first argument of `train`.
        """
        _ = kwargs
        return path

    def load_model(self, path=None):
        """ Loader for pretrained model.

        Parameters
        ----------
        path : str

        Return
        ------
        path : str
            If function is not defined in child class, return `path` to model.
            If `pretrained` is enabled, output of that function will be used
            as the second argument of `inference`.
        """
        return path

    def train(self, train_dataset, **kwargs):
        """ Function that must contain the whole training process. If `pretrained`
        is not enabled, output of that function will be used as the second argument
        of `inference`. """
        pass

    def inference(self, test_dataset, train_output, **kwargs):
        """ Function that must contain the whole inference process. The function must return
        predictions and targets for metrics.
        """
        pass

    def _compute_metrics(self):
        """ Metrics computation. """
        for _metric in self._metrics:
            if hasattr(self, _metric):
                self.metrics[_metric] = getattr(self, _metric)(self.targets, self.predictions)
            else:
                metric_config = self.config.get(_metric, {})
                metric_class = metric_config.pop('class', 'classification')
                evaluate = metric_config.pop('evaluate', {})
                metrics = METRICS[metric_class](self.targets, self.predictions, **metric_config)
                self.metrics[_metric] = metrics.evaluate(_metric, **evaluate)

    def run(self):
        """ Run validator """
        if 'train' in self.config:
            self.train_dataset = self.train_loader(**self._train_ds)
            self.from_train = self.train(self.train_dataset, **self.config['train'])
        elif 'pretrained' in self.config:
            self.from_train = self.load_model(**self._pretrained)

        if 'test' in self.config:
            self.test_dataset = self.test_loader(**self._test_ds)
            self.targets, self.predictions = self.inference(self.test_dataset, self.from_train, **self.config['test'])
            self._compute_metrics()
