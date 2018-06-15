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
    axis : int
        a class axis (default is None)
    proba : bool
        whether arrays contain probabilities (default is True)
    logits : bool
        whether arrays contain logits (default is False)
    threshold : float
        A probability level for binarization (lower values become 0, equal or greater values become 1)

    Notes
    -----
    `num_classes` and `axis` cannot be both None. If `axis` is specified, then `predictions` should be
    a one-hot array with class information provided in the given axis (class probabilities or logits).

    Due to randomness any given batch may not contain items of some classes, so all the labels cannot be
    inferred as simply as `targets.max()`.

    If both `proba` and `logits` are False, then `targets` and `predictions` are expected to contain labels.
    As a consequence, `num_classes` should be specified.

    If `proba` or `logits` is True, and `axis` is also None, then two class classification is assumed.
    """
    def __init__(self, targets, predictions, num_classes=None, axis=None, proba=True, logits=False, threshold=.5):
        self.num_classes = num_classes if num_classes is not None else 2 if axis is None else targets.shape[axis]

        if targets.ndim == predictions.ndim:
            # targets and predictions contain the same info (labels, probabilities or logits)
            targets = self._prepare(targets, proba, logits, axis, threshold)
        elif targets.ndim == predictions.ndim - 1:
            # targets contains labels while predictions is a one-hot array
            pass
        else:
            raise ValueError("targets and predictions should have compatible shapes",
                             targets.shape, predictions.shape)
        predictions = self._prepare(predictions, proba, logits, axis, threshold)
        self.classes = np.arange(self.num_classes)
        self._inferred_2class = self.num_classes == 2 and axis is None


        self._convert_to_scalar = False
        if targets.ndim == 1:
            targets = targets.reshape(1, -1)
            predictions = predictions.reshape(1, -1)
            self._convert_to_scalar = True

        self.targets = targets
        self.predictions = predictions
        self._confusion_matrix = None

        self._calc_confusion()

    def _prepare(self, arr, proba, logits, axis, threshold):
        if logits:
            arr = sigmoid(arr)
        if proba or logits:
            arr = binarize(arr, threshold)
            if axis is not None:
                arr = arr.argmax(axis=axis)
        return arr

    def _all(self):
        return [self.targets, self.predictions]

    def _items(self):
        return ([self.targets[i], self.predictions[i]] for i in range(len(self.targets)))

    def _confusion_conditions(self):
        if self._inferred_2class:
            return [[t, p] for p in (self.predictions, 1 - self.predictions) for t in (self.targets, 1 - self.targets)]
        else:
            pass

    def _gather_confusion(self, all_results):
        self._confusion_matrix = np.array(all_results).T.reshape(-1, self.num_classes, self.num_classes)

    @parallel('_confusion_conditions', post='_gather_confusion')
    @mjit
    def _calc_confusion(_, targets, predictions):
        return np.sum(targets * predictions, axis=1)

    def _return(self, value):
        return value[0] if self._convert_to_scalar else value

    def true_positive(self):
        return self._return(self._confusion_matrix[:, 0, 0])

    def false_positive(self):
        return self._return(self._confusion_matrix[:, 0, 1])

    def true_negative(self):
        return self._return(self._confusion_matrix[:, 1, 1])

    def false_negative(self):
        return self._return(self._confusion_matrix[:, 1, 0])

    def condition_positive(self):
        return self._return(self._confusion_matrix[:, :, 0].sum(axis=1))

    def condition_negative(self):
        return self._return(self._confusion_matrix[:, :, 1].sum(axis=1))

    def prediction_positive(self):
        return self._return(self._confusion_matrix[:, 0].sum(axis=1))

    def prediction_negative(self):
        return self._return(self._confusion_matrix[:, 1].sum(axis=1))

    def total_population(self):
        return self._return(self._confusion_matrix.sum(axis=(1, 2)))

    def true_positive_rate(self):
        return self.true_positive() / self.condition_positive()

    def sensitivity(self):
        return self.true_positive_rate()

    def recall(self):
        return self.true_positive_rate()

    def false_positive_rate(self):
        return self.false_positive() / self.condition_negative()

    def fallout(self):
        return self.false_positive_rate()

    def false_negative_rate(self):
        return self.false_negative() / self.condition_positive()

    def miss_rate(self):
        return self.false_negative_rate()

    def true_negative_rate(self):
        return self.true_negative() / self.condition_negative()

    def specificity(self):
        return self.true_negative_rate()

    def prevalence(self):
        return self.condition_positive() / self.total_population()

    def accuracy(self):
        return (self.true_positive() + self.true_negative()) / self.total_population()

    def positive_predictive_value(self):
        return self.true_positive() / self.prediction_positive()

    def precision(self):
        return self.positive_predictive_value()

    def false_discovery_rate(self):
        return self.false_positive() / self.prediction_positive()

    def false_omission_rate(self):
        return self.false_negative() / self.prediction_negative()

    def negative_predictive_value(self):
        return self.true_negative() / self.prediction_negative()

    def positive_likelihood_ratio(self):
        return self.true_positive_rate() / self.false_positive_rate()

    def negative_likelihood_ratio(self):
        return self.false_negative_rate() / self.true_negative_rate()

    def diagnostics_odds_ratio(self):
        return self.positive_likelihood_ratio() / self.negative_likelihood_ratio()

    def f1_score(self):
        return 2 / (1 / self.recall() + 1 / self.precision())
