""" Contains two class classification metrics """
import warnings
warnings.filterwarnings("ignore")

import numpy as np

from ... import mjit, parallel
from . import Metrics, binarize, sigmoid


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
    `num_classes` and `axis` cannot be both None. If `axis` is specified, then `predictions` should be
    a one-hot array with class information provided in the given axis (class probabilities or logits).

    If `fmt` is 'labels', `num_classes` should be specified. Due to randomness any given batch may not
    contain items of some classes, so all the labels cannot be inferred as simply as `labels.max()`.

    If `fmt` is 'proba' or 'logits', then `axis` points to the one-hot dimension.
    However, if `axis` is None, then two class classification is assumed and `targets` / `predictions`
    should contain probabilities or logits for a positive class only.

    .. note:: Count-based metrics (`true_positive`, `false_positive`, etc.) do not support aggregations.
              They always produce the output of (batch_items, num_classes) shape for multi-class tasks
              and (batch_items,) for 2-class tasks.
              For aggregations use rate metrics, such as `true_positive_rate`, `false_positive_rate`, etc.

    **Aggregation**

    In a multiclass case metrics might be calculated with or without class aggregations.

    Available aggregation are:

    - `None` - no aggregation, calculate metrics for each class individually (one-vs-all)
    - `'micro'` - calculate metrics globally by counting the total true positives,
                  false negatives, false positives, etc. across all classes
    - `'macro'` - calculate metrics for each class, and take their mean

    Examples
    --------

    ::

        m = ClassificationMetrics(targets, predictions, num_classes=10, fmt='labels')
        m.evaluate(['sensitivity', 'specificity'], agg='micro')

    """
    def __init__(self, targets, predictions, fmt='proba', num_classes=None, axis=None, threshold=.5):
        self.num_classes = num_classes if num_classes is not None else 2 if axis is None else targets.shape[axis]

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

        self._convert_to_scalar = False
        if targets.ndim == 1:
            targets = targets.reshape(1, -1)
            predictions = predictions.reshape(1, -1)
            self._convert_to_scalar = True

        self.targets = targets
        self.predictions = predictions

        self._confusion_matrix = np.zeros((self.targets.shape[0], self.num_classes, self.num_classes), dtype=np.int64)
        self._calc_confusion()

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

    def _confusion_params(self):
        self._confusion_matrix[:] = 0
        return [[self.targets, self.predictions, self.num_classes, self._confusion_matrix]]

    @parallel("_confusion_params")
    @mjit
    def _calc_confusion(self, targets, predictions, num_classes, confusion):
        for i in range(targets.shape[0]):
            targ = targets[i].flatten()
            pred = predictions[i].flatten()
            for t in range(num_classes):
                coords = np.where(targ == t)
                for c in pred[coords]:
                    confusion[i, c, t] += 1

    def _return(self, value):
        return value[0] if self._convert_to_scalar and isinstance(value, np.ndarray) else value

    def _count(self, f, label=None):
        if self.num_classes > 2:
            if label is None:
                v = np.array([self._return(f(l)) for l in range(self.num_classes)]).T
                return v
        label = 1 if label is None else label
        return self._return(f(label))

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
        return self._count(lambda l: self.total_population() - self.condition_positive(l), label)

    def prediction_positive(self, label=None, *args, **kwargs):
        _ = args, kwargs
        return self._count(lambda l: self._confusion_matrix[:, l].sum(axis=1), label)

    def prediction_negative(self, label=None, *args, **kwargs):
        _ = args, kwargs
        return self._count(lambda l: self.total_population() - self.prediction_positive(l), label)

    def total_population(self, *args, **kwargs):
        _ = args, kwargs
        return self._return(self._confusion_matrix.sum(axis=(1, 2)))

    def _calc_agg_metric(self, numer, denom, label=None, agg=None, when_zero=None):
        _when_zero = lambda n: np.where(n > 0, when_zero[0], when_zero[1])
        if self.num_classes > 2:
            label = label if label is not None else list(range(self.num_classes))
            label = label if isinstance(label, (list, tuple)) else [label]
            label_value = [(numer(l, agg=agg), denom(l, agg=agg)) for l in label]

            if agg is None:
                value = [np.where(l[1] > 0, l[0] / l[1], _when_zero(l[0])) for l in label_value]
                value = value[0] if len(value) == 1 else np.array(value).T
            if agg == 'micro':
                n = np.sum([l[0] for l in label_value], axis=0)
                d = np.sum([l[1] for l in label_value], axis=0)
                value = np.where(d > 0, n / d, _when_zero(n))
            elif agg in ['macro', 'mean']:
                value = np.mean([np.where(l[1] > 0, l[0] / l[1], _when_zero(l[0])) for l in label_value], axis=0)
        else:
            label = label if label is not None else 1
            d = denom(label)
            n = numer(label)
            value = np.where(d > 0, n / d, _when_zero(n))
        return value

    def true_positive_rate(self, label=None, agg='micro'):
        return self._calc_agg_metric(self.true_positive, self.condition_positive, label, agg, when_zero=(0, 1))

    def sensitivity(self, label=None, agg='micro'):
        return self.true_positive_rate(label, agg)

    def recall(self, label=None, agg='micro'):
        return self.true_positive_rate(label, agg)

    def false_positive_rate(self, label=None, agg='micro'):
        return self._calc_agg_metric(self.false_positive, self.condition_negative, label, agg, when_zero=(1, 0))

    def fallout(self, label=None, agg='micro'):
        return self.false_positive_rate(label, agg)

    def false_negative_rate(self, label=None, agg='micro'):
        return self._calc_agg_metric(self.false_negative, self.condition_positive, label, agg, when_zero=(1, 0))

    def miss_rate(self, label=None, agg='micro'):
        return self.false_negative_rate(label, agg)

    def true_negative_rate(self, label=None, agg='micro'):
        return self._calc_agg_metric(self.true_negative, self.condition_negative, label, agg, when_zero=(0, 1))

    def specificity(self, label=None, agg='micro'):
        return self.true_negative_rate(label, agg)

    def prevalence(self, label=None, agg='micro'):
        return self._calc_agg_metric(self.condition_positive, self.total_population, label, agg)

    def accuracy(self, *args, **kwargs):
        _ = args, kwargs
        return np.sum([self.true_positive(l) for l in range(self.num_classes)], axis=0) / self.total_population()

    def positive_predictive_value(self, label=None, agg='micro'):
        return self._calc_agg_metric(self.true_positive, self.prediction_positive, label, agg, when_zero=(0, 1))

    def precision(self, label=None, agg='micro'):
        return self.positive_predictive_value(label, agg)

    def false_discovery_rate(self, label=None, agg='micro'):
        return self._calc_agg_metric(self.false_positive, self.prediction_positive, label, agg, when_zero=(1, 0))

    def false_omission_rate(self, label=None, agg='micro'):
        return self._calc_agg_metric(self.false_negative, self.prediction_negative, label, agg, when_zero=(1, 0))

    def negative_predictive_value(self, label=None, agg='micro'):
        return self._calc_agg_metric(self.true_negative, self.prediction_negative, label, agg, when_zero=(0, 1))

    def positive_likelihood_ratio(self, label=None, agg='micro'):
        return self._calc_agg_metric(self.true_positive_rate, self.false_positive_rate, label, agg)

    def negative_likelihood_ratio(self, label=None, agg='micro'):
        return self._calc_agg_metric(self.false_negative_rate, self.true_negative_rate, label, agg)

    def diagnostics_odds_ratio(self, label=None, agg='micro'):
        return self._calc_agg_metric(self.positive_likelihood_ratio, self.negative_likelihood_ratio, label, agg)

    def f1_score(self, label=None, agg='micro'):
        return 2 / (1 / self.recall(label, agg) + 1 / self.precision(label, agg))
