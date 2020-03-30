""" Contains regression metrics """
import numpy as np

from . import Metrics

METRICS_ALIASES = {'mae': 'mean_absolute_error',
                   'mse': 'mean_squared_error',
                   'rmse': 'root_mean_squared_error',
                   'r2': 'r2_score',
                   'acc': 'accuracy'}

class RegressionMetrics(Metrics):
    """ Metrics to assess regression models.

    Available metrics:
        - mae
        - mse
        - rmse
        - r2_score
        - explained_variance_ratio
        - median absolute error
        - max error
        - accuracy

    Parameters
    ----------
    targets : array-like
        Correct target values. The shape must be (n_samples) or (n_samples, n_outputs) or list of such arrays.

    predictions : array-like
        Estimated target values. The shape must be (n_samples) or (n_samples, n_outputs) or list of such arrays.

    multi: bool
        Whether the task is multioutput (default is False).

    weights : array-like
        Sample weights. The shape must be (n_samples).

    gap : float
        Max difference between target and prediction for sample to be considered as properly classified (default is 3).

    Notes
    -----
    - For all the metrics, except max error, accuracy and median absolute error, you can compute sample-wise weighting.
    For that purpose specify `weight` argument, which must be the same size as inputs.

    Multioutput task restricted to the case where each target is 1D array.
    In the multioutput case metrics might be calculated with or without outputs averaging.

    Available methods are:

    - `None` - no averaging, calculate metrics for each output individually
    - `mean` - calculate metrics for each output, and take their mean.

    Examples
    --------
    ::

        metrics = RegressionMetrics(targets, predictions, multi=True)
        metrics.evaluate('mae', agg='mean')
        metrics.evaluate(['accuracy', 'mse'], agg=None)

    **Metrics**
    All metrics return:

    - a single value if targets are single output i.e. have shape (n_samples, ).
    - a single value if targets are multioutput i.e have shape (n_samples, n_outputs) and `agg` set to `mean`
    - a vector with (n_outputs, ) items if targets are multioutput and `agg` set to `None`
    """

    def __init__(self, targets, predictions, weights=None, multi=False):
        super().__init__()

        # if-else block bellow processes the case when the inputs and targets are list
        # of arrays-like. That's happening when we accumulate targets and predictions in the pipeline
        # via `.update(V('targets', mode='a'), value)`.
        # Since we are not interested in across batch aggregation we only need to concatenate this arrays along 0 axis
        # In this scenario we can mess with 2 cases:
        # 1. targets is a single batch of multioutput targets and has the following structure [[], [], ..].
        # 2. targets is a list of data come from couple of batches in a single output task
        # and has the same structure [[], [], ..].
        # In the first case concatenating such arrays does not make any sense. To separate such cases we introduce
        # 'multi' argument.

        if np.ndim(targets) == 0 or (np.ndim(targets) == 1 and multi):
            self.targets = np.array([targets])
            self.predictions = np.array([predictions])
        elif np.ndim(targets) == 1 or (np.ndim(targets) == 2 and multi):
            self.targets = np.array(targets)
            self.predictions = np.array(predictions)
        else:
            self.targets = np.concatenate(targets, axis=0)
            self.predictions = np.concatenate(predictions, axis=0)

        if weights is not None:
            self.weights = np.array(weights).flatten()
        else:
            self.weights = None

        self._agg_fn_dict.update(mean=lambda x: np.mean(x))

    def __getattr__(self, name):
        if name == "METRICS_ALIASES":
            raise AttributeError # See https://nedbatchelder.com/blog/201010/surprising_getattr_recursion.html
        name = METRICS_ALIASES.get(name, name)
        return object.__getattribute__(self, name)

    def mean_absolute_error(self):
        return np.average(np.abs(self.predictions - self.targets), axis=0, weights=self.weights)

    def mean_squared_error(self):
        return np.average((self.predictions - self.targets) ** 2, axis=0, weights=self.weights)

    def median_absolute_error(self):
        return np.median(np.abs(self.predictions - self.targets), axis=0)

    def max_error(self):
        return np.max(np.abs(self.predictions - self.targets), axis=0)

    def root_mean_squared_error(self):
        return np.sqrt(self.mean_squared_error())

    def r2_score(self):
        # pylint: disable=missing-docstring
        if self.weights is not None:
            weight = self.weights[:, np.newaxis]
        else:
            weight = 1
        numerator = (weight * (self.predictions - self.targets) ** 2).sum(axis=0)

        targets_avg = np.average(self.targets, axis=0, weights=self.weights)
        denominator = (weight * (self.targets - targets_avg) ** 2).sum(axis=0)
        return 1 - (numerator / denominator)

    def explained_variance_ratio(self):
        # pylint: disable=missing-docstring
        diff_avg = np.average(self.predictions - self.targets, axis=0, weights=self.weights)
        numerator = np.average((self.predictions - self.targets - diff_avg) ** 2, axis=0, weights=self.weights)

        targets_avg = np.average(self.targets, axis=0, weights=self.weights)
        denominator = np.average((self.targets - targets_avg) ** 2, axis=0, weights=self.weights)
        return 1 - (numerator / denominator)

    def accuracy(self, gap=3):
        """ Accuracy metric in the regression task can be interpreted as the ratio of samples
         for which `abs(target-predictoin) < gap`.

         Parameters
         ----------
         gap : int, default 3
            The maximum difference between pred and true values to classify sample as correct.
         """
        return (np.abs(self.predictions - self.targets) < gap).sum(axis=0) / self.targets.shape[0]
    