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
        if isinstance(self.name, _NamedExpression):
            return self.name.get(batch=batch, pipeline=pipeline, model=model)
        return self.name

    def set(self, value, batch=None, pipeline=None, model=None):
        """ Assign a value to a named expression """
        _ = value, batch, pipeline, model
        raise NotImplementedError("set should be defined in child classes")

    def append(self, value, *args, **kwargs):
        """ Append a value to a named expression
        (see list.append https://docs.python.org/3/tutorial/datastructures.html#more-on-lists) """
        self.get(*args, **kwargs).append(value)

    def extend(self, value, *args, **kwargs):
        """ Extend a value to a named expression
        (see list.extend https://docs.python.org/3/tutorial/datastructures.html#more-on-lists) """
        self.get(*args, **kwargs).extend(value)

class B(_NamedExpression):
    """ Batch component name """
    def get(self, batch=None, pipeline=None, model=None):
        """ Return a value of a batch component """
        name = super().get(batch=batch, pipeline=pipeline, model=model)
        if isinstance(batch, _DummyBatch):
            raise ValueError("Batch expressions are not allowed in static models B(%s)" % name)
        return getattr(batch, name)

    def set(self, value, batch=None, pipeline=None, model=None):
        """ Assign a value to a batch component """
        name = super().get(batch=batch, pipeline=pipeline, model=model)
        setattr(batch, name, value)


class C(_NamedExpression):
    """ A pipeline config option """
    def get(self, batch=None, pipeline=None, model=None):
        """ Return a value of a pipeline config """
        name = super().get(batch=batch, pipeline=pipeline, model=model)
        pipeline = batch.pipeline if batch is not None else pipeline
        config = pipeline.config or {}
        return config.get(name, None)

    def set(self, value, batch=None, pipeline=None, model=None):
        """ Assign a value to a pipeline config """
        name = super().get(batch=batch, pipeline=pipeline, model=model)
        pipeline = batch.pipeline if batch is not None else pipeline
        config = pipeline.config or {}
        config[name] = value


class F(_NamedExpression):
    """ A function, method or any other callable """
    def get(self, batch=None, pipeline=None, model=None):
        """ Return a value from a callable """
        name = super().get(batch=batch, pipeline=pipeline, model=model)
        args = []
        if isinstance(batch, _DummyBatch) or batch is None:
            _pipeline = batch.pipeline if batch is not None else pipeline
            args += [_pipeline]
        else:
            args += [batch]
        if model is not None:
            args += [model]
        return name(*args)

    def set(self, *args, **kwargs):
        """ Assign a value by calling a callable """
        _ = args, kwargs
        raise NotImplementedError("Setting a value with a callable is not supported")


class V(_NamedExpression):
    """ Pipeline variable name """
    def get(self, batch=None, pipeline=None, model=None):
        """ Return a value of a pipeline variable """
        name = super().get(batch=batch, pipeline=pipeline, model=model)
        pipeline = batch.pipeline if batch is not None else pipeline
        return pipeline.get_variable(name)

    def set(self, value, batch=None, pipeline=None, model=None):
        """ Assign a value to a pipeline variable """
        name = super().get(batch=batch, pipeline=pipeline, model=model)
        pipeline = batch.pipeline if batch is not None else pipeline
        pipeline.set_variable(name, value)
