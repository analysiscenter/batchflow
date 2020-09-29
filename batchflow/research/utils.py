""" Auxilary functions """


def get_metrics(pipeline, metrics_var, metrics_name, *args, agg='mean', **kwargs):
    """ Function to evaluate metrics """
    metrics_name = metrics_name if isinstance(metrics_name, list) else [metrics_name]
    metrics = pipeline.get_variable(metrics_var).evaluate(metrics_name, *args, agg=agg, **kwargs)
    values = [metrics[name] for name in metrics_name]
    if len(values) == 1:
        return values[0]
    return values
