""" Auxilary functions """


def get_metrics(pipeline, metrics_var, metrics_name, *args, agg='mean', **kwargs):
    """ Function to evaluate metrics """
    metrics = pipeline.get_variable(metrics_var)
    return metrics.evaluate(metrics_name, *args, agg=agg, **kwargs)
