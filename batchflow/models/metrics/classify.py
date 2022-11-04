""" Contains two class classification metrics """
from copy import copy
from functools import partial

import numpy as np

from ...decorators import mjit
from . import Metrics, binarize, sigmoid, infmean
from ...plotter import plot



METRICS_ALIASES = {'sensitivity' : 'true_positive_rate',
                   'recall' : 'true_positive_rate',
                   'tpr' : 'true_positive_rate',
                   'fallout' : 'false_positive_rate',
                   'fpr' : 'false_positive_rate',
                   'miss_rate' : 'false_negative_rate',
                   'fnr' : 'false_negative_rate',
                   'specificity' : 'true_negative_rate',
                   'tnr' : 'true_negative_rate',
                   'prv' : 'prevalence',
                   'acc' : 'accuracy',
                   'precision' : 'positive_predictive_value',
                   'ppv' : 'positive_predictive_value',
                   'fdr' : 'false_discovery_rate',
                   'for' : 'false_omission_rate',
                   'npv' : 'negative_predictive_value',
                   'plr' : 'positive_likelihood_ratio',
                   'nlr' : 'negative_likelihood_ratio',
                   'dor' : 'diagnostics_odds_ratio',
                   'dice' : 'f1_score',
                   'f1s' : 'f1_score',
                   'iou' : 'jaccard',
                   'jac' : 'jaccard'}

class ClassificationMetrics(Metrics):
    """ Metrics to assess classification models

    Parameters
    ----------
    targets : np.array
        Ground-truth labels / probabilities / logits
    predictions : np.array
        Predicted labels / probabilites / logits
    num_classes : int
        the number of classes (default is None)
    fmt : 'proba', 'logits', 'labels'
        whether arrays contain probabilities, logits or labels
    axis : int
        a class axis (default is None)
    threshold : float
        A probability level for binarization (lower values become 0, equal or greater values become 1)

    Notes
    -----
    - Input arrays (`targets` and `predictions`) might be vectors or multidimensional arrays,
      where the first dimension represents batch items. The latter is useful for pixel-level metrics.

    - Both `targets` and `predictions` usually contain the same data (labels, probabilities or logits).
      However, `targets` might be labels, while `predictions` are probabilities / logits.
      For that to work:

      - `targets` should have the shape which exactly 1 dimension smaller, than `predictions` shape;
      - `axis` should point to that dimension;
      - `fmt` should contain format of `predictions`.

    - When `axis` is specified, `predictions` should be a one-hot array with class information provided
      in the given axis (class probabilities or logits). In this case `targets` can contain labels (see above)
      or probabilities / logits in the very same axis.

    - If `fmt` is 'labels', `num_classes` should be specified. Due to randomness any given batch may not
      contain items of some classes, so all the labels cannot be inferred as simply as `labels.max()`.

    - If `fmt` is 'proba' or 'logits', then `axis` points to the one-hot dimension.
      However, if `axis` is None, two class classification is assumed and `targets` / `predictions`
      should contain probabilities or logits for a positive class only.


    **Metrics**
    All metrics return:

    - a single value if input is a vector for a 2-class task.
    - a single value if input is a vector for a multiclass task and multiclass averaging is enabled.
    - a vector with batch size items if input is a multidimensional array (e.g. images or sequences)
      and there are just 2 classes or multiclass averaging is on.
    - a vector with `num_classes` items if input is a vector for multiclass case without averaging.
    - a 2d array `(batch_items, num_classes)` for multidimensional inputs in a multiclass case without averaging.

    .. note:: Count-based metrics (`true_positive`, `false_positive`, etc.) do not support mutliclass averaging.
              They always return counts for each class separately.
              For multiclass tasks rate metrics, such as `true_positive_rate`, `false_positive_rate`, etc.,
              might seem more convenient.

    **Multiclass metrics**

    In a multiclass case metrics might be calculated with or without class averaging.

    Available methods are:

    - `None` - no averaging, calculate metrics for each class individually (one-vs-all)
    - `'micro'` - calculate metrics globally by counting the total true positives,
                  false negatives, false positives, etc. across all classes
    - `'macro'` - calculate metrics for each class, and take their mean.


    Examples
    --------

    ::

        metrics = ClassificationMetrics(targets, predictions, num_classes=10, fmt='labels')
        metrics.evaluate(['sensitivity', 'specificity'], multiclass='macro')

    """
    def __init__(self, targets, predictions, fmt='proba', num_classes=None, axis=None, threshold=.5,
                 skip_bg=False, calc=True):
        super().__init__()
        self.targets = None
        self.predictions = None
        self._confusion_matrix = None
        self.skip_bg = skip_bg
        self.num_classes = None if axis is None else predictions.shape[axis]
        self.num_classes = self.num_classes or num_classes or 2
        self._agg_fn_dict = {'mean': partial(infmean, axis=0)}

        if fmt in ['proba', 'logits'] and axis is None and self.num_classes > 2:
            raise ValueError('axis cannot be None for multiclass case when fmt is proba or logits')

        if targets.ndim == predictions.ndim:
            # targets and predictions contain the same info (labels, probabilities or logits)
            targets = self._to_labels(targets, fmt, axis, threshold)
        elif targets.ndim == predictions.ndim - 1 and fmt != 'labels':
            # targets contains labels while predictions is a one-hot array
            pass
        else:
            raise ValueError("targets and predictions should have compatible shapes",
                             targets.shape, predictions.shape)
        predictions = self._to_labels(predictions, fmt, axis, threshold)

        if targets.ndim == 1:
            targets = targets.reshape(1, -1)
            predictions = predictions.reshape(1, -1)
            self._no_zero_axis = True
        else:
            self._no_zero_axis = False

        self.targets = targets
        self.predictions = predictions

        if calc:
            self._calc()

    def __getattr__(self, name):
        if name == "METRICS_ALIASES":
            raise AttributeError # See https://nedbatchelder.com/blog/201010/surprising_getattr_recursion.html
        name = METRICS_ALIASES.get(name, name)
        return object.__getattribute__(self, name)

    @property
    def confusion_matrix(self):
        return self._confusion_matrix.sum(axis=0)

    def plot_confusion_matrix(self, classes=None, normalize=False, **kwargs):
        """ Plot confusion matrix.

        Parameters
        ----------
        classes : sequence, optional
            Sequence of classes labels.
        normalize : bool
            Whether to normalize confusion matrix over target classes.
        """
        confusion_matrix = np.array(self.confusion_matrix)
        if classes is None:
            classes = np.arange(self.num_classes)
        if normalize:
            confusion_matrix = confusion_matrix / np.nansum(confusion_matrix, axis=0)

        plot_config = {
            'title': 'Normalized confusion matrix' if normalize else 'Confusion matrix',
            'xlabel': 'Actual class',
            'xtick_locations': np.arange(confusion_matrix.shape[0]),
            'xtick_labels': classes,
            'xtick_rotation': 90,
            'xtick_ha': 'center',
            'ylabel': 'Predicted class',
            'ytick_locations': np.arange(confusion_matrix.shape[1]),
            'ytick_labels': classes,
            'ytick_rotation': 0,
            'ytick_va': 'center',
            **kwargs
        }

        return plot(data=confusion_matrix, mode='matrix', **plot_config)

    def copy(self):
        """ Return a duplicate containing only the confusion matrix """
        metrics = copy(self)
        metrics.free()
        return metrics

    def _to_labels(self, arr, fmt, axis, threshold):
        if fmt == 'labels':
            pass
        elif fmt in ['proba', 'logits']:
            if axis is None:
                if fmt == 'logits':
                    arr = sigmoid(arr)
                arr = binarize(arr, threshold).astype('int8')
            else:
                arr = arr.argmax(axis=axis)
        return arr

    def one_hot(self, inputs):
        """ Convert an array of labels into a one-hot array """
        return np.eye(self.num_classes)[inputs] if self.num_classes > 2 else inputs

    def free(self):
        """ Free memory allocated for intermediate data """
        self.targets = None
        self.predictions = None

    def append(self, metrics):
        """ Append confusion matrix with data from another metrics"""
        # pylint: disable=protected-access
        self._confusion_matrix = np.concatenate((self._confusion_matrix, metrics._confusion_matrix), axis=0)

    def update(self, metrics):
        """ Update confusion matrix with data from another metrics"""
        # pylint: disable=protected-access
        if self._no_zero_axis:
            self._confusion_matrix = self._confusion_matrix + metrics._confusion_matrix
        else:
            self._confusion_matrix = np.concatenate((self._confusion_matrix, metrics._confusion_matrix), axis=0)

    def __getitem__(self, item):
        # pylint: disable=protected-access
        metrics = self.copy()
        metrics._confusion_matrix = metrics._confusion_matrix[item]
        return metrics

    def _calc(self):
        self._confusion_matrix = np.zeros((self.targets.shape[0], self.num_classes, self.num_classes), dtype=np.intp)
        return self._calc_confusion_jit(self.targets, self.predictions, self.num_classes, self._confusion_matrix)

    @mjit
    def _calc_confusion_jit(self, targets, predictions, num_classes, confusion):
        for i in range(targets.shape[0]):
            targ = targets[i].flatten()
            pred = predictions[i].flatten()
            for t in range(num_classes):
                coords = np.where(targ == t)
                for c in pred[coords]:
                    confusion[i, c, t] += 1

    def _return(self, value):
        return value[0] if isinstance(value, np.ndarray) and value.shape == (1, ) else value

    def _all_labels(self):
        first = 1 if self.skip_bg else 0
        labels = list(range(first, self.num_classes))
        return labels

    def _count(self, f, label=None):
        if label is None:
            label = self._all_labels() if self.num_classes > 2 else 1
        if np.isscalar(label):
            return self._return(f(label))
        return np.array([self._return(f(l)) for l in label]).T

    def true_positive(self, label=None, *args, **kwargs):
        _ = args, kwargs
        return self._count(lambda l: self._confusion_matrix[:, l, l], label)

    def false_positive(self, label=None, *args, **kwargs):
        _ = args, kwargs
        return self._count(lambda l: self.prediction_positive(l) - self.true_positive(l), label)

    def true_negative(self, label=None, *args, **kwargs):
        _ = args, kwargs
        return self._count(lambda l: self.condition_negative(l) - self.false_positive(l), label)

    def false_negative(self, label=None, *args, **kwargs):
        _ = args, kwargs
        return self._count(lambda l: self.condition_positive(l) - self.true_positive(l), label)

    def condition_positive(self, label=None, *args, **kwargs):
        _ = args, kwargs
        return self._count(lambda l: self._confusion_matrix[:, :, l].sum(axis=1), label)

    def condition_negative(self, label=None, *args, **kwargs):
        _ = args, kwargs
        return self._count(lambda l: self.total_population(l) - self.condition_positive(l), label)

    def prediction_positive(self, label=None, *args, **kwargs):
        _ = args, kwargs
        return self._count(lambda l: self._confusion_matrix[:, l].sum(axis=1), label)

    def prediction_negative(self, label=None, *args, **kwargs):
        _ = args, kwargs
        return self._count(lambda l: self.total_population(l) - self.prediction_positive(l), label)

    def total_population(self, *args, **kwargs):
        _ = args, kwargs
        return self._return(self._confusion_matrix.sum(axis=(1, 2)))

    def _calc_agg(self, numer, denom, label=None, multiclass='macro', when_zero=None):
        _when_zero = lambda n: np.where(n > 0, when_zero[0], when_zero[1]).astype(float)
        if self.num_classes == 2:
            label = label if label is not None else 1
        labels = label if label is not None else self._all_labels()
        labels = labels if isinstance(labels, (list, tuple)) else [labels]
        fractions = [(numer(l).astype(float), denom(l).astype(float)) for l in labels]

        if multiclass is None:
            value = [np.divide(n, d, out=_when_zero(n), where=(d > 0)).ravel() for n, d in fractions]
            classes_calculated = self.num_classes - 1 if self.skip_bg else self.num_classes
            value = value[0] if len(value) == 1 else np.array(value).T.reshape(-1, classes_calculated)
        elif multiclass == 'micro':
            n = np.sum([f[0] for f in fractions], axis=0)
            d = np.sum([f[1] for f in fractions], axis=0)
            value = np.divide(n, d, out=_when_zero(n), where=(d > 0)).reshape(-1, 1)
        elif multiclass in ['macro', 'mean']:
            value = [np.divide(n, d, out=_when_zero(n), where=(d > 0)) for n, d in fractions]
            value = infmean(value, axis=0).reshape(-1, 1)

        return value

    def true_positive_rate(self, *args, when_zero=(0, 1), **kwargs):
        return self._calc_agg(self.true_positive, self.condition_positive, *args, when_zero=when_zero, **kwargs)

    def false_positive_rate(self, *args, when_zero=(1, 0), **kwargs):
        return self._calc_agg(self.false_positive, self.condition_negative, *args, when_zero=when_zero, **kwargs)

    def false_negative_rate(self, *args, when_zero=(1, 0), **kwargs):
        return self._calc_agg(self.false_negative, self.condition_positive, *args, when_zero=when_zero, **kwargs)

    def true_negative_rate(self, *args, when_zero=(0, 1), **kwargs):
        return self._calc_agg(self.true_negative, self.condition_negative, *args, when_zero=when_zero, **kwargs)

    def prevalence(self, *args, when_zero=(0, 0), **kwargs):
        """
        Notes
        -----
        Parameter when_zero doesn't really matter in this case,
        since total_population is never zero, when targets are not empty.
        """
        return self._calc_agg(self.condition_positive, self.total_population, *args, when_zero=when_zero, **kwargs)

    def accuracy(self):
        """ An accuracy of detecting all the classes combined """
        return np.sum([self.true_positive(l) for l in self._all_labels()], axis=0) / self.total_population()

    def positive_predictive_value(self, *args, when_zero=(0, 1), **kwargs):
        return self._calc_agg(self.true_positive, self.prediction_positive, *args, when_zero=when_zero, **kwargs)

    def false_discovery_rate(self, *args, when_zero=(1, 0), **kwargs):
        return self._calc_agg(self.false_positive, self.prediction_positive, *args, when_zero=when_zero, **kwargs)

    def false_omission_rate(self, *args, when_zero=(1, 0), **kwargs):
        return self._calc_agg(self.false_negative, self.prediction_negative, *args, when_zero=when_zero, **kwargs)

    def negative_predictive_value(self, *args, when_zero=(0, 1), **kwargs):
        return self._calc_agg(self.true_negative, self.prediction_negative, *args, when_zero=when_zero, **kwargs)

    def positive_likelihood_ratio(self, *args, when_zero=(np.inf, 0), **kwargs):
        return self._calc_agg(self.true_positive_rate, self.false_positive_rate, *args, when_zero=when_zero, **kwargs)

    def negative_likelihood_ratio(self, *args, when_zero=(np.inf, 0), **kwargs):
        return self._calc_agg(self.false_negative_rate, self.true_negative_rate, *args, when_zero=when_zero, **kwargs)

    def diagnostics_odds_ratio(self, *args, when_zero=(np.inf, 0), **kwargs):
        return self._calc_agg(self.positive_likelihood_ratio, self.negative_likelihood_ratio,
                              *args, when_zero=when_zero, **kwargs)

    def f1_score(self, *args, **kwargs):
        """ Compute f1-score """
        recall = self.recall(*args, when_zero=(0, np.inf), **kwargs)
        precision = self.precision(*args, when_zero=(0, np.inf), **kwargs)
        mask = np.isinf(recall) & np.isinf(precision)
        value = np.nan_to_num(2 * (recall * precision) / (recall + precision))
        value[mask] = np.inf
        return value

    def jaccard(self, *args, **kwargs):
        d = self.dice(*args, **kwargs)
        return np.nan_to_num(d / (2 - d), nan=np.inf)
