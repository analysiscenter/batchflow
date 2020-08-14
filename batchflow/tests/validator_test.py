# pylint: disable=redefined-outer-name, missing-docstring
import pytest
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

from batchflow.models.validator import Validator
from batchflow import Config

@pytest.fixture
def dummy_validator_class():
    class DummyValidator(Validator):
        def train(self, a):
            if a is None:
                a = 2
            self.a = a

        def inference(self, b):
            if b is None:
                b = 1
            return self.a, b

        def my_metric(self, a, b):
            return a - b
    return DummyValidator


@pytest.fixture
def validator_class(cv=False):
    class NewValidator(Validator):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.model = None
            self.stages = []

        def load_train_dataset(self, size=100):
            self.stages += ['load_train_dataset']

            np.random.seed(42)
            x = np.random.random(size).reshape(-1, 1)
            self.train_dataset = x, 2 * x
            return self.train_dataset

        def load_test_dataset(self, size=50):
            self.stages += ['load_test_dataset']

            np.random.seed(6)
            x = np.random.random(size).reshape(-1, 1)
            self.test_dataset = x, 2 * x
            return self.test_dataset

        def load_cv_dataset(self, size=100, folds=5):
            self.stages += ['load_cv_dataset']
            datasets = []
            x = np.random.random(size).reshape(-1, 1)
            y = 2 * x
            for train_indices, test_indices in KFold(n_splits=folds).split(x):
                datasets.append([
                    (x[train_indices], y[train_indices]),
                    (x[test_indices], y[test_indices]),
                ])
            return datasets

        def train(self, dataset, model='linear'):
            self.stages += ['train']

            if model == 'linear':
                model = LinearRegression()
            elif model == 'lasso':
                model = Lasso()
            model.fit(dataset[0], dataset[1])
            self.model = model

        def load_model(self):
            self.stages += ['load_model']

            model = LinearRegression()
            model.coef_ = np.array([2])
            model.intercept_ = 0
            self.model = model

        def inference(self, dataset):
            self.stages += ['inference']

            targets = dataset[1]
            if self.model:
                predictions = self.model.predict(dataset[0])
            else:
                predictions = None
            return targets, predictions

        def my_mse(self, target, prediction):
            self.stages += ['my_mse']

            return mean_squared_error(target, prediction)
    if cv:
        class NewValidator(NewValidator):
            def load_train_test_dataset(self, **kwargs):
                train = self.load_train_dataset()
                test = self.load_test_dataset()
                return train, test

    return NewValidator

@pytest.mark.parametrize('config, metric', [
    (Config({'train_dataset/path': 5, 'test_dataset/path': 2, 'metrics': 'my_metric'}), 3),
    (Config({'test_dataset/path': 2, 'metrics': 'my_metric'}), 0),
    (Config({'train_dataset/path': 3, 'metrics': 'my_metric'}), 2),
    (Config({'metrics': 'my_metric'}), 1),
])
def test_args(dummy_validator_class, config, metric):
    val = dummy_validator_class(config)
    val.run()
    assert val.metrics['my_metric'] == metric

@pytest.mark.parametrize('config, stages', [
    ({}, ['train', 'inference', 'load_train_dataset', 'load_test_dataset']),
    ({'train': {}, 'test': {}}, ['train', 'inference', 'load_train_dataset', 'load_test_dataset']),
    ({'pretrained': {}, 'test': {}}, ['load_model', 'inference', 'load_test_dataset']),
    ({'train': {}}, ['train', 'inference', 'load_train_dataset', 'load_test_dataset']),
    ({'test': {}}, ['train', 'inference', 'load_train_dataset', 'load_test_dataset']),
    ({'load_train_dataset': {}}, ['train', 'inference', 'load_train_dataset', 'load_test_dataset']),
    ({'train': {}, 'metrics': 'my_mse'}, ['train', 'inference', 'load_train_dataset', 'load_test_dataset', 'my_mse']),
    ({'cv': None}, ['train', 'inference', 'load_cv_dataset'])
])
def test_stages(validator_class, config, stages):
    val = validator_class(config)
    val.run()
    assert set(val.stages) == set(stages)

@pytest.mark.parametrize('config', [
    Config({'train_dataset/size': 100, 'train/model': 'lasso', 'test_dataset/size': 10})
])
def test_kwargs(validator_class, config):
    val = validator_class(config)
    val.run()
    assert len(val.train_dataset[1]) == config['train_dataset/size']
    assert len(val.test_dataset[1]) == config['test_dataset/size']
    assert isinstance(val.model, Lasso)

@pytest.mark.parametrize('config', [
    Config({'metrics': 'my_mse, mse', 'mse/class': 'regression'})
])
def test_metrics(validator_class, config):
    val = validator_class(config)
    val.run()
    assert np.isclose(val.metrics['mse'], 0)
    assert np.isclose(val.metrics['my_mse'], 0)

@pytest.mark.parametrize('add_methods, methods, exception', [
    (('train', 'inference'), ('train', 'inference'), False),
    (('train', 'inference'), ['train'], False),
    (('inference', ), ['train', 'inference'], True),
    (('load_model', 'inference'), ['train', 'inference'], True),
    (('load_model', 'inference'), ['train|load_model', 'inference'], False),
    (('train',), ('train',), False),
    ({'test': None}, ['train', 'load_model'], True)
])
def test_api_checks(add_methods, methods, exception):
    val_class = type('NewValidator', (Validator, ), {method: lambda x: x for method in add_methods})
    assert val_class.check_api(methods=methods, warning=False) == exception

@pytest.mark.parametrize('folds, agg', list(zip([3, 5], ['mean', 'median'])))
def test_cv(validator_class, folds, agg):
    val = validator_class(Config({'metrics': 'my_mse', 'cv': {'folds': folds, 'agg': agg}}))
    val.run()
    assert np.isclose(val.metrics['my_mse'], 0)

@pytest.mark.parametrize('folds', [3, 5])
def test_cv_none_agg(validator_class, folds):
    val = validator_class(Config({'metrics': 'my_mse', 'cv': {'folds': folds, 'agg': None}}))
    val.run()
    assert len(val.metrics) == folds
