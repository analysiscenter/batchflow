""" Contains metrics for segmentation """
import numpy as np

from . import ClassificationMetrics, get_components


class SegmentationMetricsByPixels(ClassificationMetrics):
    """ Metrics to assess segmentation models pixel-wise """
    pass


class SegmentationMetricsByInstances(ClassificationMetrics):
    """ Metrics to assess segmentation models by instances (connected components) """
    def __init__(self, targets, predictions, fmt='proba', num_classes=None, axis=None, threshold=.5, iot=.5):
        super().__init__(targets, predictions, fmt, num_classes, axis, threshold, confusion=False)
        self.iot = iot
        self.target_instances = self._get_instances(self.targets, axis)
        self.predicted_instances = self._get_instances(self.predictions, axis)
        self._calc_confusion()

    def _get_instances(self, inputs, axis):
        if axis is None:
            instances = [get_components(inputs, batch=True)]
        else:
            instances = []
            shape = [slice(None)] * inputs.ndim
            for i in range(1, inputs.shape[axis]):
                shape[axis] = i
                per_class = get_components(inputs[shape], batch=True)
                instances.append(per_class)
        return instances

    def _calc_confusion(self):
        self._confusion_matrix = np.zeros((self.targets.shape[0], self.num_classes - 1, 2, 2), dtype=np.intp)

        for k in range(self.num_classes - 1):
            for i in range(len(self.target_instances[k])):
                for j in range(len(self.target_instances[k][i])):
                    c = self.target_instances[k][i][j]
                    coords = [slice(i, i+1)]
                    for d in range(c.shape[0]):
                        coords.append(slice(c[d, 0], c[d, 1]))
                    targ = self.targets[coords]
                    pred = self.targets[coords] * self.predictions[coords]
                    if np.sum(pred) / targ.size >= self.iot:
                        self._confusion_matrix[i, k, 1, 1] += 1
                    else:
                        self._confusion_matrix[i, k, 0, 1] += 1

        for k in range(self.num_classes - 1):
            for i in range(len(self.predicted_instances[k])):
                for j in range(len(self.predicted_instances[k][i])):
                    c = self.predicted_instances[k][i][j]
                    coords = [slice(i, i+1)]
                    for d in range(c.shape[0]):
                        coords.append(slice(c[d, 0], c[d, 1]))
                    pred = self.predictions[coords]
                    targ = self.targets[coords]
                    if np.sum(pred) / targ.size < self.iot:
                        self._confusion_matrix[i, k, 1, 0] += 1

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
