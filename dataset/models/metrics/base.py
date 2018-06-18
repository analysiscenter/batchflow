""" Contains a base metrics class """

class Metrics:
    """ Base metrics evalation class """
    def evaluate(self, metrics, *args, **kwargs):
        """ Calculates metrics

        Parameters
        ----------
        metrics : list of str
            metrics names

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
