""" Contains a base metrics class """


class Metrics:
    """ Base metrics evalation class

    Examples
    --------

    ::

    m = ClassificationMetrics(targets, predictions, num_classes=10, fmt='labels')
    m.evaluate(['sensitivity', 'specificity'], agg='micro')
    """
    def evaluate(self, metrics, *args, **kwargs):
        """ Calculates metrics

        Parameters
        ----------
        metrics : list of str
            metrics names
        args, kwargs
            metric-specific parameters

        Returns
        -------
        dict
            key - metric name
            value - metric value
        """
        res = {}
        for name in metrics:
            metric_fn = getattr(self, name)
            res[name] = metric_fn(*args, **kwargs)
        return res
