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
    name : str
        name of the current model (from yaml)
    task : str
        name of the current task (from yaml)
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
            self._train_ds = self.config['train'].pop('dataset', {})
            if isinstance(self._train_ds, str):
                self._train_ds = {'path': self._train_ds}

        self._test_ds = {}
        if 'test' in self.config and self.config['test'] is not None:
            self._test_ds = self.config['test'].pop('dataset', {})
            if isinstance(self._test_ds, str):
                self._test_ds = {'path': self._test_ds}
            self._metrics = self.config['test'].pop('metrics', {})
            self._custom_metrics = self.config['test'].pop('custom_metrics', [])

        self.train_dataset = None
        self.from_train = None
        self.test_dataset = None
        self.targets = None
        self.predictions = None

        self.custom_metric_values = {}
        self.metric_values = {}

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

    def _compute_custom_metrics(self):
        self.custom_metric_values = {}
        for _metric in self._custom_metrics:
            value = getattr(self, _metric)(self.targets, self.predictions)
            self.custom_metric_values[_metric] = value

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
            self._compute_custom_metrics()



class Validator:
    """ Validator class

    **Attributes**

    config_path : str
        path to yaml file with config of the following structure:
        ```
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
        ```

    **How to get metrics values**

        After execution Validator will have attribute `results` with the resulting ModelAPI instances.
        To get metrics, call `.metrics` or `.custom_metrics` properties.
    """
    def __init__(self, config_path='validator.yaml'):
        self.config_path = config_path
        with open(self.config_path) as file:
            self.config = yaml.load(file, Loader=yaml.Loader)
        self.results = {}

    def start(self):
        """ Start validator """
        import model_api #pylint: disable=unused-import,import-error
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
