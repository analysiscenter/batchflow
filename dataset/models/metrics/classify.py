""" Contains two class classification metrics """
import numpy as np

from ... import mjit, parallel
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

    def _all(self):
        return [self.targets, self.predictions]

    def _items(self):
        return ([self.targets[i], self.predictions[i]] for i in range(len(self.targets)))

    def _confusion_params(self):
        self._confusion_matrix = np.zeros((self.targets.shape[0], self.num_classes, self.num_classes), dtype=np.int64)
        return [[self.targets, self.predictions, t, self._confusion_matrix] for t in range(self.num_classes)]

    @parallel('_confusion_params')
    @mjit
    def _calc_confusion(_, targets, predictions, target_class, confusion):
        coords = np.where(targets == target_class)
        pred_classes = predictions[coords]
        for i in range(len(coords[0])):
            confusion[coords[0][i]][pred_classes[i], target_class] += 1

    def _return(self, value):
        return value[0] if self._convert_to_scalar else value

    def true_positive(self, label=0):
        return self._return(self._confusion_matrix[:, label, label])

    def false_positive(self, label=0):
        return self.prediction_positive(label) - self.true_positive(label)

    def true_negative(self, label=0):
        return self.condition_negative(label) - self.false_positive(label)

    def false_negative(self, label=0):
        return self.condition_positive(label) - self.true_positive(label)

    def condition_positive(self, label=0):
        return self._return(self._confusion_matrix[:, :, label].sum(axis=1))

    def condition_negative(self, label=0):
        return self.total_population() - self.condition_positive(label)

    def prediction_positive(self, label=0):
        return self._return(self._confusion_matrix[:, label].sum(axis=1))

    def prediction_negative(self, label=0):
        return self.total_population() - self.prediction_positive(label)

    def total_population(self):
        return self._return(self._confusion_matrix.sum(axis=(1, 2)))

    def true_positive_rate(self, label=0):
        return self.true_positive(label) / self.condition_positive(label)

    def sensitivity(self, label=0):
        return self.true_positive_rate(label)

    def recall(self, label=0):
        return self.true_positive_rate(label)

    def false_positive_rate(self, label=0):
        return self.false_positive(label) / self.condition_negative(label)

    def fallout(self, label=0):
        return self.false_positive_rate(label)

    def false_negative_rate(self, label=0):
        return self.false_negative(label) / self.condition_positive(label)

    def miss_rate(self, label=0):
        return self.false_negative_rate(label)

    def true_negative_rate(self, label=0):
        return self.true_negative(label) / self.condition_negative(label)

    def specificity(self, label=0):
        return self.true_negative_rate(label)

    def prevalence(self, label=0):
        return self.condition_positive(label) / self.total_population()

    def accuracy(self, label=0):
        return (self.true_positive(label) + self.true_negative(label)) / self.total_population()

    def positive_predictive_value(self, label=0):
        return self.true_positive(label) / self.prediction_positive(label)

    def precision(self, label=0):
        return self.positive_predictive_value(label)

    def false_discovery_rate(self, label=0):
        return self.false_positive(label) / self.prediction_positive(label)

    def false_omission_rate(self, label=0):
        return self.false_negative(label) / self.prediction_negative(label)

    def negative_predictive_value(self, label=0):
        return self.true_negative(label) / self.prediction_negative(label)

    def positive_likelihood_ratio(self, label=0):
        return self.true_positive_rate(label) / self.false_positive_rate(label)

    def negative_likelihood_ratio(self, label=0):
        return self.false_negative_rate(label) / self.true_negative_rate(label)

    def diagnostics_odds_ratio(self, label=0):
        return self.positive_likelihood_ratio(label) / self.negative_likelihood_ratio(label)

    def f1_score(self, label=0):
        return 2 / (1 / self.recall(label) + 1 / self.precision(label))
