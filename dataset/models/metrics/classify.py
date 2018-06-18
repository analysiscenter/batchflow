""" Contains two class classification metrics """
import numpy as np

from . import Metrics, binarize, sigmoid


class ClassificationMetrics(Metrics):
    """ Metrics used for classification models

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

        self._confusion_matrix = None
        self._calc_confusion()

    def _to_labels(self, arr, fmt, axis, threshold):
        if fmt == 'labels':
            pass
        elif fmt in ['proba', 'logits']:
            if axis is None:
                if fmt == 'logits':
                    arr = sigmoid(arr)
                arr = binarize(arr, threshold)
            else:
                arr = arr.argmax(axis=axis)
        return arr

    def _calc_confusion(self):
        self._confusion_matrix = np.zeros((self.targets.shape[0], self.num_classes, self.num_classes), dtype=np.int64)
        for t in range(self.num_classes):
            coords = np.where(self.targets == t)
            pred_classes = self.predictions[coords]
            for i in range(len(coords[0])):
                item = self._confusion_matrix[coords[0][i]]
                item[pred_classes[i], t] += 1

    def _return(self, value):
        return value[0] if self._convert_to_scalar else value

    def true_positive(self, label=0, *args, **kwargs):
        _ = args, kwargs
        return self._return(self._confusion_matrix[:, label, label])

    def false_positive(self, label=0, *args, **kwargs):
        _ = args, kwargs
        return self.prediction_positive(label) - self.true_positive(label)

    def true_negative(self, label=0, *args, **kwargs):
        _ = args, kwargs
        return self.condition_negative(label) - self.false_positive(label)

    def false_negative(self, label=0, *args, **kwargs):
        _ = args, kwargs
        return self.condition_positive(label) - self.true_positive(label)

    def condition_positive(self, label=0, *args, **kwargs):
        _ = args, kwargs
        return self._return(self._confusion_matrix[:, :, label].sum(axis=1))

    def condition_negative(self, label=0, *args, **kwargs):
        _ = args, kwargs
        return self.total_population() - self.condition_positive(label)

    def prediction_positive(self, label=0, *args, **kwargs):
        _ = args, kwargs
        return self._return(self._confusion_matrix[:, label].sum(axis=1))

    def prediction_negative(self, label=0, *args, **kwargs):
        _ = args, kwargs
        return self.total_population() - self.prediction_positive(label)

    def total_population(self, *args, **kwargs):
        _ = args, kwargs
        return self._return(self._confusion_matrix.sum(axis=(1, 2)))

    def _calc_agg_metric(self, numer, denom, label=None, agg=None):
        if self.num_classes > 2:
            label = label if label is not None else list(range(self.num_classes))
            label = label if isinstance(label, (list, tuple)) else [label]
            label_value = [(numer(l, agg=agg), denom(l, agg=agg)) for l in label]

            if agg is None:
                value = [l[0] / l[1] for l in label_value]
                value = value[0] if len(value) == 1 else np.array(value).T
            if agg == 'micro':
                value = np.sum([l[0] for l in label_value], axis=0) / np.sum([l[1] for l in label_value], axis=0)
            elif agg in ['macro', 'mean']:
                value = np.mean([l[0] / l[1] for l in label_value], axis=0)
        else:
            label = label if label is not None else 0
            value = numer(label) / denom(label)
        return value

    def true_positive_rate(self, label=None, agg='micro'):
        return self._calc_agg_metric(self.true_positive, self.condition_positive, label, agg)

    def sensitivity(self, label=None, agg='micro'):
        return self.true_positive_rate(label, agg)

    def recall(self, label=None, agg='micro'):
        return self.true_positive_rate(label, agg)

    def false_positive_rate(self, label=None, agg='micro'):
        return self._calc_agg_metric(self.false_positive, self.condition_negative, label, agg)

    def fallout(self, label=None, agg='micro'):
        return self.false_positive_rate(label, agg)

    def false_negative_rate(self, label=None, agg='micro'):
        return self._calc_agg_metric(self.false_negative, self.condition_positive, label, agg)

    def miss_rate(self, label=None, agg='micro'):
        return self.false_negative_rate(label, agg)

    def true_negative_rate(self, label=None, agg='micro'):
        return self._calc_agg_metric(self.true_negative, self.condition_negative, label, agg)

    def specificity(self, label=None, agg='micro'):
        return self.true_negative_rate(label, agg)

    def prevalence(self, label=None, agg='micro'):
        return self._calc_agg_metric(self.condition_positive, self.total_population, label, agg)

    def accuracy(self, *args, **kwargs):
        _ = args, kwargs
        return np.sum([self.true_positive(l) for l in range(self.num_classes)], axis=0) / self.total_population()

    def positive_predictive_value(self, label=None, agg='micro'):
        return self._calc_agg_metric(self.true_positive, self.prediction_positive, label, agg)

    def precision(self, label=None, agg='micro'):
        return self.positive_predictive_value(label, agg)

    def false_discovery_rate(self, label=None, agg='micro'):
        return self._calc_agg_metric(self.false_positive, self.prediction_positive, label, agg)

    def false_omission_rate(self, label=None, agg='micro'):
        return self._calc_agg_metric(self.false_negative, self.prediction_negative, label, agg)

    def negative_predictive_value(self, label=None, agg='micro'):
        return self._calc_agg_metric(self.true_negative, self.prediction_negative, label, agg)

    def positive_likelihood_ratio(self, label=None, agg='micro'):
        return self._calc_agg_metric(self.true_positive_rate, self.false_positive_rate, label, agg)

    def negative_likelihood_ratio(self, label=None, agg='micro'):
        return self._calc_agg_metric(self.false_negative_rate, self.true_negative_rate, label, agg)

    def diagnostics_odds_ratio(self, label=None, agg='micro'):
        return self._calc_agg_metric(self.positive_likelihood_ratio, self.negative_likelihood_ratio, label, agg)

    def f1_score(self, label=None, agg='micro'):
        return 2 / (1 / self.recall(label, agg) + 1 / self.precision(label, agg))
