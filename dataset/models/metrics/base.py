""" Contains a base metrics class """

class Metrics:
    def evaluate(self, metrics):
        res = {}
        for name in metrics:
            metric_fn = getattr(self, name)
            res[name] = metric_fn()
        return res
