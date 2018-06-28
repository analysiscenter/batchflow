""" Contains metrics for segmentation """
import numpy as np

from . import ClassificationMetrics, get_components


class SegmentationMetricsByPixels(ClassificationMetrics):
    """ Metrics to assess segmentation models pixel-wise """
    pass


class SegmentationMetricsByInstances(ClassificationMetrics):
    """ Metrics to assess segmentation models by instances (connected components)

    Parameters
    ----------
    iot : float
        if the ratio of a predicted instance size to the corresponding target size >= `iot`,
        then instance is considered correctly predicted (true postitive).

    Notes
    -----
    For other parameters see :class:`~.ClassificationMetrics`.

    """
    def __init__(self, targets, predictions, fmt='proba', num_classes=None, axis=None,
                 skip_bg=True, threshold=.5, iot=.5):
        super().__init__(targets, predictions, fmt, num_classes, axis, threshold, skip_bg, confusion=False)
        self.iot = iot
        self.target_instances = self._get_instances(self.one_hot(self.targets), axis)
        self.predicted_instances = self._get_instances(self.one_hot(self.predictions), axis)
        self._calc_confusion()
        self.target_instances = None
        self.predicted_instances = None

    def _get_instances(self, inputs, axis):
        """ Find instances of each class within inputs

        Parameters
        ----------
        inputs : np.ndarray
            one-hot array
        axis : int
            a class axis

        Returns
        -------
        nested list with ndarray of coords
            num_classes - 1, batch_items, num_instances, inputs.shape, number of pixels
        """
        if axis is None:
            instances = [get_components(inputs, batch=True)]
        else:
            instances = []
            shape = [slice(None)] * inputs.ndim
            for i in range(1, inputs.shape[axis]):
                shape[axis] = i
                one_class = get_components(inputs[shape], batch=True)
                instances.append(one_class)
        return instances

    def _calc_confusion(self):
        self._confusion_matrix = np.zeros((self.targets.shape[0], self.num_classes - 1, 2, 2), dtype=np.intp)

        for k in range(1, self.num_classes):
            for i, item_instances in enumerate(self.target_instances[k-1]):
                for coords in item_instances:
                    targ = len(coords[0])
                    pred = np.sum(self.predictions[i][coords] == k)
                    if np.sum(pred) / targ >= self.iot:
                        self._confusion_matrix[i, k-1, 1, 1] += 1
                    else:
                        self._confusion_matrix[i, k-1, 0, 1] += 1

        for k in range(1, self.num_classes):
            for i, item_instances in enumerate(self.predicted_instances[k-1]):
                for coords in item_instances:
                    pred = len(coords[0])
                    targ = np.sum(self.targets[i][coords] == k)
                    if targ == 0 or pred / targ < self.iot:
                        self._confusion_matrix[i, k-1, 1, 0] += 1

    def true_positive(self, label=None, *args, **kwargs):
        _ = args, kwargs
        return self._count(lambda l: self._confusion_matrix[:, l-1, 1, 1], label)

    def true_negative(self, label=None, *args, **kwargs):
        raise ValueError("True negative is inapplicable for instance-based metrics")

    def condition_positive(self, label=None, *args, **kwargs):
        _ = args, kwargs
        return self._count(lambda l: self._confusion_matrix[:, l-1, :, 1].sum(axis=1), label)

    def prediction_positive(self, label=None, *args, **kwargs):
        _ = args, kwargs
        return self._count(lambda l: self._confusion_matrix[:, l-1, 1].sum(axis=1), label)

    def total_population(self, label=None, *args, **kwargs):
        _ = args, kwargs
        return self._return(self._confusion_matrix[:, label - 1].sum(axis=(1, 2)))
