""" Contains metrics for segmentation """
from ... import parallel
from . import ClassificationMetrics, get_components


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
        self._calc_confusion()

    def _calc_confusion(self):
        for i in range(len(self.target_components)):
            for j in range(len(self.target_components[i])):
                pass
