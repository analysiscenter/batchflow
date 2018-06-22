""" Contains metrics for segmentation """
import numpy as np

from ... import parallel
from . import ClassificationMetrics, get_components
from time import time


class SegmentationMetricsByPixels(ClassificationMetrics):
    """ Metrics to assess segmentation models pixel-wise """
    pass


class SegmentationMetricsByComponents(ClassificationMetrics):
    """ Metrics to assess segmentation models with connected components """
    def __init__(self, targets, predictions, fmt='proba', num_classes=None, axis=None, threshold=.5, iot=.5):
        super().__init__(targets, predictions, fmt, num_classes, axis, threshold, confusion=False)
        self.iot = iot
        self.target_components = get_components(self.targets, batch=True)
        self.predicted_components = get_components(self.predictions, batch=True)
        t = time()
        self._calc_confusion()
        print(time() - t)

    def _calc_confusion(self):
        self._confusion_matrix = np.zeros((self.targets.shape[0], self.num_classes - 1, 2, 2), dtype=np.int32)
        print(self.target_components)
        print(self.predicted_components)

        for i in range(len(self.target_components)):
            for j in range(len(self.target_components[i])):
                for k in range(self.num_classes - 1):
                    c = self.target_components[i][j]
                    coords = [slice(i, i+1)]
                    for d in range(c.shape[0]):
                        coords.append(slice(c[d, 0], c[d, 1]))
                    print(coords)
                    t = self.targets[coords]
                    p = self.predictions[coords]
                    #self._confusion_matrix[i, k, 1, 1] += 1
