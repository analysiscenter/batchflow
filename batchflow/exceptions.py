""" Contains specific Exceptions """


class BatchFlowException(Exception): # noqa: N818; error-suffix-on-exception-name
    """ Base exception class """

class SkipBatchException(BatchFlowException):
    """ Throw this in an action-method if you want to skip the batch from the rest of the pipeline """

class StopPipeline(BatchFlowException):
    """ Stop the entire pipeline run. """

class EmptyBatchSequence(BatchFlowException, Warning):
    """ Throw this when batch generator is empty """
