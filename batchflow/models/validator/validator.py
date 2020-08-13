""" Wrapper for models """

import warnings
import inspect
from copy import deepcopy
import yaml
import numpy as np

from ...pipeline import METRICS
from ... import Config

def _get_method_owner(meth):
    if inspect.ismethod(meth):
        for cls in inspect.getmro(meth.__self__.__class__):
            if cls.__dict__.get(meth.__name__) is meth:
                return cls
        meth = meth.__func__  # fallback to __qualname__ parsing
    if inspect.isfunction(meth):
        cls = getattr(inspect.getmodule(meth),
                      meth.__qualname__.split('.<locals>', 1)[0].rsplit('.', 1)[0])
        if isinstance(cls, type):
            return cls
    return getattr(meth, '__objclass__', None)  # handle special descriptor objects

class Validator:
    """ Validator: interface for loaders, train, inference and metrics evaluation.

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
    def __init__(self, config=None, **kwargs):
        if isinstance(config, str):
            self.config_path = config
            with open(config) as file:
                self.config = yaml.load(file, Loader=yaml.Loader)
        elif isinstance(config, dict):
            self.config_path = None
            self.config = deepcopy(config)
        elif isinstance(config, Config):
            self.config_path = None
            self.config = deepcopy(config.config)
        elif config is None:
            self.config = {}


        defaults = {
            'pretrained': {},
            'train': {},
            'test': {},
            'train_dataset': {},
            'test_dataset': {},
            'metrics': [],
            'cv': {}
        }

        self.config = {**defaults, **self.config, **kwargs}

        if 'train' in self.config and 'pretrained' in self.config:
            raise ValueError("Both 'train' and 'pretrained' was found.")

        for key in ['pretrained', 'train_dataset', 'test_dataset', 'cv']:
            if self.config[key] is None:
                self.config[key] = {}
            elif isinstance(self.config[key], str):
                self.config[key] = {'path': self.config[key]}

        if isinstance(self.config['metrics'], str):
            self.config['metrics'] = [item.strip() for item in self.config['metrics'].split(',')]

        self.targets = None
        self.predictions = None

        self.metrics = {}

    def run(self):
        """ Run validator """
        if 'cv' not in self.config:
            self._run()
        else:
            self._run_cv()

    def _run(self):
        loader_kwargs = self.config.get('load_train_test_dataset') or {}
        train_dataset, test_dataset = self.load_train_test_dataset(**loader_kwargs)

        if 'pretrained' in self.config:
            self.load_model(**self._pretrained)
        else:
            train_dataset = train_dataset or self.load_train_dataset(**self.config['train_dataset'])
            self.train(train_dataset, **self.config['train'])

        test_dataset = test_dataset or self.load_test_dataset(**self.config['test_dataset'])
        self.targets, self.predictions = self.inference(test_dataset, **self.config['test'])

        metrics_kwargs = {key: value for key, value in self.config.items() if key in self.config['metrics']}
        self.metrics = self.compute_metrics(self.targets, self.predictions, *self.config['metrics'], **metrics_kwargs)

    def _run_cv(self):
        res = []
        metrics_kwargs = {key: value for key, value in self.config.items() if key in self.config['metrics']}
        agg = self.config['cv'].pop('agg', 'mean')
        for train, test in self.load_cv_dataset(**self.config['cv']):
            self.train(train, **self.config['train'])
            self.targets, self.predictions = self.inference(test, **self.config['test'])
            res.append(self.compute_metrics(self.targets, self.predictions, *self.config['metrics'], **metrics_kwargs))
        if agg is None:
            self.metrics = res
        else:
            if isinstance(agg, str):
                _agg = getattr(np, agg)
            elif callable(agg):
                _agg = agg
            else:
                raise ValueError('agg must be str or callable')
            for key in res[0]:
                self.metrics[key] = _agg([item[key] for item in res])

    def load_train_dataset(self, path=None, **kwargs):
        """ Train dataset loader.

        Parameters
        ----------
        path : str or None
            path to train dataset, defined in `train/dataset` section of config

        Return
        ------
        path : str or None
            If function is not defined in child class, return `path`. It will be
            used as first argument of `train`.
         """
        _ = kwargs
        return path

    def load_test_dataset(self, path=None, **kwargs):
        """ Test dataset loader.

        Parameters
        ----------
        path : str or None
            path to test dataset, defined in `test/dataset` section of config

        Return
        ------
        path : str or None
            If function is not defined in child class, return `path`. It will be
            used as the first argument of `train`.
        """
        _ = kwargs
        return path

    def load_train_test_dataset(self, path=None, **kwargs):
        """ Train and test datasets loader.

        Parameters
        ----------
        path : str or None
            path to data

        Return
        ------
        tuple :
            If function is not defined in child class, return (`path`, None). It will be
            used as the first arguments of `train` and `test` correspondingly. For each None
            value in tuple corresponding loader will be called (train or test).
        """
        _ = kwargs
        return path, None

    def load_cv_dataset(self, path, **kwargs):
        """ Cross valdiation split.

        Parameters
        ----------
        path : str or None
            path to data

        Return
        ------
        list of tuples :
            Each tuple is pair of train and test datasets. By default, return empty list.
        """
        _ = path, kwargs
        return []

    def load_model(self, path=None):
        """ Loader for pretrained model.

        Parameters
        ----------
        path : str
            path to model

        Return
        ------
        path : str
            If function is not defined in child class, return `path` to model.
            If `pretrained` is enabled in config, 'load_train_dataset' and 'train'
            will be ignored.
        """
        _ = path

    def train(self, train_dataset, **kwargs):
        """ Function that must contain the whole training process. Method will be executed
        if `pretrained` is not enabled. """
        pass

    def inference(self, test_dataset, **kwargs):
        """ Function that must contain the whole inference process. The function must return
        predictions and targets for metrics.
        """
        pass

    def compute_metrics(self, targets, predictions, *metrics, **metric_configs):
        """ Metrics computation. """
        metrics = list(set([*metrics, *metric_configs.keys()]))
        values = {}
        for _metric in metrics:
            if hasattr(self, _metric):
                values[_metric] = getattr(self, _metric)(targets, predictions)
            else:
                metric_config = metric_configs.get(_metric, {})
                metric_class = metric_config.pop('class', 'classification')
                evaluate = metric_config.pop('evaluate', {})
                metrics = METRICS[metric_class](targets, predictions, **metric_config)
                values[_metric] = metrics.evaluate(_metric, **evaluate)
        return values

    @classmethod
    def check_api(cls, methods=('train', 'inference'), warning=True, exception=False):
        """ Check that Validator child class implements necessary methods.

        Parameters
        ----------
        methods : list
            list of methods to check.
        warning : bool
            if True, call warning.
        exception : bool
            if True, raise exception.

        Returns
        -------
        error: bool
        """
        error = False
        if isinstance(methods, str):
            methods = (methods, )
        if exception:
            def _warning(msg):
                raise NotImplementedError(msg)
        elif warning:
            _warning = warnings.warn
        else:
            _warning = lambda msg: msg
        for meth in methods:
            cond = all([_get_method_owner(getattr(cls, _meth)) == Validator for _meth in meth.split('|')])
            if cond:
                error = True
                _ = _warning('Method "{}" is not implemented in class {}'.format(meth, cls.__class__))
        return error

    def check_config(self, keys=('train', 'test'), warning=True, exception=False):
        """ Check that config has necessary keys.

        Parameters
        ----------
        keys : list
            list of keys to check.
        warning : bool
            if True, call warning. If exception
        exception : bool
            if True, raise exception.

        Returns
        -------
        error: bool
        """
        error = False
        if isinstance(keys, str):
            keys = (keys, )
        if exception:
            def _warning(msg):
                raise NotImplementedError(msg)
        elif warning:
            _warning = warnings.warn
        else:
            _warning = lambda msg: msg
        for key in keys:
            cond = all([_key not in self.config for _key in key.split('|')])
            if cond:
                error = True
                _ = _warning('Key "{}" was not found in config: {}'.format(key, self.config))
        return error
