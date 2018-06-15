""" Contains two class classification metrics """
import numpy as np

from ... import mjit, parallel
from . import Metrics, binarize
#from .utils import binarize


class ClassificationMetrics(Metrics):
    """ Metrics used for 2-class classification and semantic segmentation models

    Parameters
    ----------
    targets : np.array
        Ground-truth labels or probabilities
    predictions : np.array
        Predicted probabilites for a positive class (labeled as 1)
    bin : bool or None
        whether to binarize targets and predictions (default is True)
    threshold : float
        A binarization level (lower values become 0, equal or greater values become 1)
    """
    def __init__(self, targets, predictions, bin=True, threshold=.5):
        if targets.ndim != predictions.ndim:
            raise ValueError("targets and predictions should have similar shapes.")
        if targets.ndim not in (1, 2):
            raise ValueError("Arrays should be 1- or 2-dimensional.")
        self._reshape = False
        if targets.ndim == 1:
            targets = targets.reshape(1, -1)
            self._reshape = True

        self._confusion_matrix = None
        self.threshold = threshold
        if bin:
            self.targets = binarize(targets, threshold)
            self.predictions = binarize(predictions, threshold)
        else:
            self.targets = targets
            self.predictions = predictions

        self._calc_confusion()


    def _all(self):
        return [self.targets, self.predictions]

    def _items(self):
        return ([self.targets[i], self.predictions[i]] for i in range(len(self.targets)))

    def _confusion_conditions(self):
        return [[t, p] for p in (self.predictions, 1 - self.predictions) for t in (self.targets, 1 - self.targets)]

    def _gather_confusion(self, all_results):
        self._confusion_matrix = np.array(all_results).T.reshape(-1, 2, 2)

    @parallel('_confusion_conditions', post='_gather_confusion')
    @mjit
    def _calc_confusion(_, targets, predictions):
        return np.sum(targets * predictions, axis=1)

    def _return(self, value):
        return value[0] if self._reshape else value

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
