""" Auxilary functions """

def get_metrics(pipeline, metrics_var, metrics_name):
    """ Function to evaluate metrics """
    metrics = pipeline.get_variable(metrics_var)
    return metrics.evaluate(metrics_name)
