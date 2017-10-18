""" Contains named expression classes"""


class _DummyBatch:
    """ A fake batch for static models """
    def __init__(self, pipeline):
        self.pipeline = pipeline


class _NamedExpression:
    """ Base class for named expression """
    def __init__(self, name):
        self.name = name

    def get(self, batch=None, pipeline=None, model=None):
        """ Return a value of a named expression """
        _ = batch, pipeline, model
        raise NotImplementedError("get should be defined in child classes")

    def set(self, value, batch=None, pipeline=None, model=None):
        """ Assign a value to a named expression """
        _ = value, batch, pipeline, model
        raise NotImplementedError("set should be defined in child classes")

    def append(self, value, *args, **kwargs):
        """ Append a value to a named expression """
        self.get(*args, **kwargs).append(value)


class B(_NamedExpression):
    """ Batch component name """
    def get(self, batch=None, pipeline=None, model=None):
        """ Return a value of a batch component """
        _ = pipeline, model
        if isinstance(batch, _DummyBatch):
            raise ValueError("Batch expressions are not allowed in static models B(%s)" % self.name)
        return getattr(batch, self.name)

    def set(self, value, batch=None, pipeline=None, model=None):
        """ Assign a value to a batch component """
        _ = pipeline, model
        setattr(batch, self.name, value)


class C(_NamedExpression):
    """ Callable """
    def get(self, batch=None, pipeline=None, model=None):
        """ Return a value from a callable """
        args = []
        if isinstance(batch, _DummyBatch):
            args += [batch.pipeline or pipeline]
        else:
            args += [batch]
        if model is not None:
            args += [model]
        return self.name(*args)

    def set(self, *args, **kwargs):
        """ Assign a value by calling a callable """
        _ = args, kwargs
        raise NotImplementedError("Setting a value with a callable is not supported")

    def append(self, *args, **kwargs):
        """ Append a value by calling a callable """
        _ = args, kwargs
        raise NotImplementedError("Appending with a callable is not supported")


class V(_NamedExpression):
    """ Pipeline variable name """
    def get(self, batch=None, pipeline=None, model=None):
        """ Return a value of a pipeline variable """
        _ = model
        pipeline = batch.pipeline or pipeline
        return pipeline.get_variable(self.name)

    def set(self, value, batch=None, pipeline=None, model=None):
        """ Assign a value to a pipeline variable """
        _ = model
        pipeline = batch.pipeline or pipeline
        pipeline.set_variable(self.name, value)
