""" Contains specific Exceptions """


class BatchFlowException(Exception):
    """ Base exception class """
    pass

class SkipBatchException(BatchFlowException):
    """ Throw this in an action-method if you want to skip the batch from the rest of the pipeline """
    pass

class EmptyBatchSequence(BatchFlowException, Warning):
    """ Throw this when batch generator is empty """
    pass
