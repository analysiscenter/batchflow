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
def validator_class():
    class NewValidator(Validator):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.model = None
            self.stages = []

        def load_train_dataset(self, size=100):
            self.stages += ['train_loader']

            np.random.seed(42)
            x = np.random.random(size).reshape(-1, 1)
            return x, 2 * x

        def load_test_dataset(self, size=50):
            self.stages += ['test_loader']

            np.random.seed(6)
            x = np.random.random(size).reshape(-1, 1)
            return x, 2 * x

        def load_cv_dataset(self, size=100, folds=5):
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

    return NewValidator

@pytest.mark.parametrize('config, metric', [
    (Config({'train/dataset/path': 5, 'test/dataset/path': 2, 'test/metrics': 'my_metric'}), 3),
    (Config({'train': None, 'test/dataset/path': 2, 'test/metrics': 'my_metric'}), 0),
    (Config({'train/dataset/path': 3, 'test/metrics': 'my_metric'}), 2),
    (Config({'train': None, 'test/metrics': 'my_metric'}), 1),
])
def test_skipped_loaders(dummy_validator_class, config, metric):
    val = dummy_validator_class(config)
    val.run()
    assert val.metrics['my_metric'] == metric

@pytest.mark.parametrize('config, stages', [
    ({'train': None, 'test': None}, ['train', 'inference', 'train_loader', 'test_loader']),
    ({'pretrained': None, 'test': None}, ['load_model', 'inference', 'test_loader']),
    ({'train': None}, ['train', 'train_loader']),
    ({'train': None, 'test': {'metrics': 'my_mse'}}, ['train', 'inference', 'train_loader', 'test_loader', 'my_mse']),
    ({'test': None}, ['test_loader', 'inference'])
])
def test_stages(validator_class, config, stages):
    val = validator_class(config)
    val.run()
    assert set(val.stages) == set(stages)

@pytest.mark.parametrize('config', [
    Config({'train/dataset/size': 100, 'train/model': 'lasso', 'test/dataset/size': 10})
])
def test_kwargs(validator_class, config):
    val = validator_class(config)
    val.run()
    assert len(val.train_dataset[1]) == config['train/dataset/size']
    assert len(val.test_dataset[1]) == config['test/dataset/size']
    assert isinstance(val.model, Lasso)

@pytest.mark.parametrize('config', [
    Config({'train': None, 'test/metrics': 'my_mse, mse', 'mse/class': 'regression'})
])
def test_metrics(validator_class, config):
    val = validator_class(config)
    val.run()
    assert np.isclose(val.metrics['mse'], 0)
    assert np.isclose(val.metrics['my_mse'], 0)

@pytest.mark.parametrize('config, keys, exception', [
    ({'train': None, 'test': None}, ['train', 'test'], False),
    ({'train': None, 'test': None}, ['train'], False),
    ({'test': None}, ['train', 'test'], True),
    ({'pretrained': None, 'test': None}, ['train', 'test'], True),
    ({'pretrained': None, 'test': None}, ['train|pretrained', 'test'], False),
    ({'train': None}, 'train', False),
    ({'test': None}, ['train', 'pretrained'], True)
])
def test_config_checks(validator_class, config, keys, exception):
    assert validator_class(config).check_config(keys=keys, warning=False) == exception

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
    val = validator_class(Config({'train': None, 'test/metrics': 'my_mse', 'cv': {'folds': folds}}))
    val.run_cv(agg=agg)
    assert np.isclose(val.metrics['my_mse'], 0)

@pytest.mark.parametrize('folds', [3, 5])
def test_cv_none_agg(validator_class, folds):
    val = validator_class(Config({'train': None, 'test/metrics': 'my_mse', 'cv': {'folds': folds}}))
    val.run_cv(agg=None)
    assert len(val.metrics) == folds
