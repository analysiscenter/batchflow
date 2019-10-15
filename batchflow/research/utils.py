""" Auxilary functions """

def get_metrics(experiment, pipeline, metrics_var, metrics_name):
    """ Function to evaluate metrics """
    pipeline = experiment[pipeline].pipeline
    metrics = pipeline.get_variable(metrics_var)
    return metrics.evaluate(metrics_name)
