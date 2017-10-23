""" Contains models for sci-kit learn estimators """
# pylint: disable=arguments-differ

try:
    from sklearn.external import joblib as pickle
except:
    pass
try:
    import dill as pickle
except:
    pass
from .base import BaseModel


class SklearnModel(BaseModel):
 """ Base class for scikit-learn models

    Attributes
    ----------
    estimator - an instance of scikit-learn estimator

    Configuration
    -------------
    estimator - an instance of scikit-learn estimator

    Examples
    --------
    >>> pipeline
            .init_model('static', SklearnModel, 'my_model',
                        config={'estimator': sklearn.linear_model.SGDClassifier(loss='huber')})
    """
    def __init__(self, *args, **kwargs):
        self.estimator = None
        super().__init__(*args, **kwargs)

    def build(self, *args, **kwargs):
        """ Define the model """
        _ = args, kwargs
        self.estimator = self.get_from_config('estimator')

    def load(self, path, *args, **kwargs):
        """ Load the model.

        Parameters
        ----------
        path : str - a full path to a file from which a model will be loaded
        """
        if self.estimator is not None:
            self.estimator = pickle.load(path)

    def save(self, path, *args, **kwargs):
        """ Save the model.

        Parameters
        ----------
        path : str - a full path to a file where a model will be saved to
        """
        if self.estimator is not None:
            pickle.dump(self.estimator, path)
        else:
            raise ValueError("Scikit-learn estimator does not exist. Check your config for 'estimator'.")

    def train(self, X, y, classes=None, sample_weight=None, *args, **kwargs):
        """ Train the model with the data provided

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Subset of the training data

        y : numpy array, shape (n_samples,)
            Subset of the target values

        classes : array, shape (n_classes,)
            Classes across all calls to partial_fit.

        sample_weight : array-like, shape (n_samples,), optional
            Weights applied to individual samples

        For more details look at the documentation for the estimator used.
        """
        self.estimator.partial_fit(X, y, classes, sample_weight)

    def predict(self, X, *args, **kwargs):
        """ Predict with the data provided

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Subset of the training data

        For more details look at the documentation for the estimator used.

        Returns
        -------
        array, shape (n_samples,)
            Predicted value per sample.
        """
        return self.estimator.predict(X)
