""" Contains a base metrics class """
import numpy as np


class Metrics:
    """ Base metrics evalation class

    This class is not supposed to be instantiated.
    Use specific children classes instead (e.g. :class:`.ClassificationMetrics`).

    Examples
    --------

    ::

        m = ClassificationMetrics(targets, predictions, num_classes=10, fmt='labels')
        m.evaluate(['sensitivity', 'specificity'], multiclass='micro')
    """
    def evaluate(self, metrics, *args, **kwargs):
        """ Calculates metrics

        Parameters
        ----------
        metrics : str or list of str
            metric names
        args
            metric-specific parameters
        kwargs
            metric-specific parameters

        Returns
        -------
        metric value or dict

            if metrics is a list, then a dict is returned::
            - key - metric name
            - value - metric value
        """
        _metrics = [metrics] if isinstance(metrics, str) else metrics

        res = {}
        for name in _metrics:
            metric_fn = getattr(self, name)
            res[name] = metric_fn(*args, **kwargs)
        res = res[metrics] if isinstance(metrics, str) else res

        return res
