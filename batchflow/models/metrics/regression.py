""" Contains regression metrics """
import numpy as np

from . import Metrics

METRICS_ALIASES = {'mae': 'mean_absolute_error',
                   'mse': 'mean_squar_error',
                   'rmse': 'root_mean_square_error',
                   'r2': 'r2_score',
                   'acc': 'accuracy'}

class RegressionMetrics(Metrics):
    """ Metrics to assess regression models

    Parameters
    ----------
    targets : array-like of shape = (n_samples) or (n_samples, n_outputs) \
              or list of such arrays
        Ground truth (correct) target values.

    predictions : array-like of shape = (n_samples) or (n_samples, n_outputs) \
                  or list of such arrays
        Estimated target values

    multi: bool
        Whether the targets are multi ouput (default is False)

    weights : array-like of shape (n_samples)
        Sample weights.

    gap : float
        Max difference between target and prediction for sample to be \
        considered as properly classified (default is 3)

    Notes
    -----
    - Accuracy metric for regression task stands for the ratio of samples for which `abs(targets-predictions) < gap`.

    - For all the metrics, except max_error and accuracy you can compute sample-wise weighing.
    For that purpose specify `weight` argument, which must be the same size as inputs.

    In a multioutput case metrics might be calculated with or without outputs averaging.

    Available methods are:

    - `None` - no averaging, calculate metrics for each output individually
    - `mean` - calculate metrics for each output, and take their mean.

    **Metrics**
    All metrics return:

    - a single value if targets are single output i.e. have shape (n_samples, ).
    - a single value if targets are multi output i.e have shape (n_samples, n_outputs) and `agg` set to `mean`
    - a vector with (n_outputs, ) items if targets are multi output and `agg` set to `None`
    """

    def __init__(self, targets, predictions, weights=None, gap=3, multi=False):
        super().__init__()

        # if-else block bellow process the case when the inputs and targets are list
        # of arrays-like. Thats happening when we accumulating targets and predictions in the pipeline
        # via `.update(V('targets', mode='a'), value)`.
        # Since we are not interested in across batch aggregation we only need to concatenate this arrays along 0 axis
        # In this scenario tho we can mess with 2 cases:
        # 1. targets is a single batch of multioutput targets and has the following structure [[], [], ..].
        # 2. targets is a list of data come from couple of batches in a single output task
        # and has the same structure [[], [], ..].
        # In the first case concatenating such arrays do not have any sence. To separate such cases we introduce
        # 'multi' argument.

        if np.ndim(targets) == 1 or (np.ndim(targets) == 2 and multi):
            self.targets = np.array(targets)
            self.predictions = np.array(predictions)
        else:
            self.targets = np.concatenate(targets, axis=0)
            self.predictions = np.concatenate(predictions, axis=0)

        self.weights = np.array(weights).flatten()
        self.gap = gap
        self.multi = multi

        self._agg_fn_dict = {
            'mean': lambda x: np.mean(x),
        }

    def __getattr__(self, name):
        if name == "METRICS_ALIASES":
            raise AttributeError # See https://nedbatchelder.com/blog/201010/surprising_getattr_recursion.html
        name = METRICS_ALIASES.get(name, name)
        return object.__getattribute__(self, name)

    def mean_absolute_error(self):
        return np.average(np.abs(self.predictions - self.targets), axis=0,
                          weights=self.weights)

    def mean_squared_error(self):
        return np.average((self.predictions - self.targets) ** 2, axis=0,
                          weights=self.weights)

    def median_absilute_error(self):
        return np.median(np.abs(self.predictions - self.targets))

    def max_error(self):
        return np.max(np.abs(self.predictions - self.targets))

    def root_mean_square_error(self):
        return np.sqrt(self.mean_squared_error())

    def r2_score(self):
        """ r2_score """
        if self.weights is not None:
            weight = self.weights[:, np.newaxis]
        else:
            weight = 1
        numerator = (weight * (self.predictions - self.targets) ** 2).sum(axis=0)
        targets_avg = np.average(self.targets, axis=0, weights=self.weights)
        denominator = (weight * (self.targets - targets_avg) ** 2).sum(axis=0)
        return 1 - (numerator / denominator)

    def explained_variance_ratio(self):
        """ explained_variance """
        diff_avg = np.average(self.predictions - self.targets, axis=0, weights=self.weights)
        numerator = np.average(self.predictions - self.targets - diff_avg, axis=0, weights=self.weights)
        targets_avg = np.average(self.targets, axis=0, weights=self.weights)
        denominator = np.average((self.targets - targets_avg) ** 2, axis=0, weights=self.weights)
        return 1 - (numerator / denominator)

    def accuracy(self):
        return np.sum(np.abs(self.predictions - self.targets) < self.gap, axis=0) \
               / self.targes.shape[0]
    