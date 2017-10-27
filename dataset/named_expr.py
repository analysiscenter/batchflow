""" Contains named expression classes"""


class _DummyBatch:
    """ A fake batch for static models """
    def __init__(self, pipeline):
        self.pipeline = pipeline


class NamedExpression:
    """ Base class for a named expression """
    def __init__(self, name):
        self.name = name

    def get(self, batch=None, pipeline=None, model=None):
        """ Return a value of a named expression """
        if isinstance(self.name, NamedExpression):
            return self.name.get(batch=batch, pipeline=pipeline, model=model)
        return self.name

    def set(self, value, batch=None, pipeline=None, model=None, mode='w'):
        """ Set a value to a named expression """
        if mode in ['a', 'append']:
            self.append(value, batch=batch, pipeline=pipeline, model=model)
        elif mode in ['e', 'extend']:
            self.extend(value, batch=batch, pipeline=pipeline, model=model)
        elif mode in ['u', 'update']:
            self.update(value, batch=batch, pipeline=pipeline, model=model)
        else:
            self.assign(value, batch=batch, pipeline=pipeline, model=model)

    def assign(self, value, batch=None, pipeline=None, model=None):
        """ Assign a value to a named expression """
        raise NotImplementedError("assign should be implemented in child classes")

    def append(self, value, *args, **kwargs):
        """ Append a value to a named expression

        if a named expression is a dict or set, `update` is called, or `append` otherwise.

        See also
        --------
        list.append https://docs.python.org/3/tutorial/datastructures.html#more-on-lists
        dict.update https://docs.python.org/3/library/stdtypes.html#dict.update
        set.update https://docs.python.org/3/library/stdtypes.html#frozenset.update
        """
        var = self.get(*args, **kwargs)
        if isinstance(var, (set, dict)):
            var.update(value)
        else:
            var.append(value)

    def extend(self, value, *args, **kwargs):
        """ Extend a named expression with a new value
        (see list.extend https://docs.python.org/3/tutorial/datastructures.html#more-on-lists) """
        self.get(*args, **kwargs).extend(value)

    def update(self, value, *args, **kwargs):
        """ Update a named expression with a new value
        (see dict.update https://docs.python.org/3/library/stdtypes.html#dict.update
        or set.update https://docs.python.org/3/library/stdtypes.html#frozenset.update) """
        self.get(*args, **kwargs).update(value)


class B(NamedExpression):
    """ Batch component name """
    def get(self, batch=None, pipeline=None, model=None):
        """ Return a value of a batch component """
        name = super().get(batch=batch, pipeline=pipeline, model=model)
        if isinstance(batch, _DummyBatch):
            raise ValueError("Batch expressions are not allowed in static models B(%s)" % name)
        return getattr(batch, name)

    def assign(self, value, batch=None, pipeline=None, model=None):
        """ Assign a value to a batch component """
        name = super().get(batch=batch, pipeline=pipeline, model=model)
        setattr(batch, name, value)


class C(NamedExpression):
    """ A pipeline config option """
    def get(self, batch=None, pipeline=None, model=None):
        """ Return a value of a pipeline config """
        name = super().get(batch=batch, pipeline=pipeline, model=model)
        pipeline = batch.pipeline if batch is not None else pipeline
        config = pipeline.config or {}
        return config.get(name, None)

    def assign(self, value, batch=None, pipeline=None, model=None):
        """ Assign a value to a pipeline config """
        name = super().get(batch=batch, pipeline=pipeline, model=model)
        pipeline = batch.pipeline if batch is not None else pipeline
        config = pipeline.config or {}
        config[name] = value


class F(NamedExpression):
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

    def assign(self, *args, **kwargs):
        """ Assign a value by calling a callable """
        _ = args, kwargs
        raise NotImplementedError("Assigning a value with a callable is not supported")


class V(NamedExpression):
    """ Pipeline variable name """
    def get(self, batch=None, pipeline=None, model=None):
        """ Return a value of a pipeline variable """
        name = super().get(batch=batch, pipeline=pipeline, model=model)
        pipeline = batch.pipeline if batch is not None else pipeline
        return pipeline.get_variable(name)

    def assign(self, value, batch=None, pipeline=None, model=None):
        """ Assign a value to a pipeline variable """
        name = super().get(batch=batch, pipeline=pipeline, model=model)
        pipeline = batch.pipeline if batch is not None else pipeline
        pipeline.set_variable(name, value)


def eval_expr(expr, batch, model=None):
    """ Evaluate a named expression recursively """
    if isinstance(expr, NamedExpression):
        expr = expr.get(batch=batch, model=model)
    elif isinstance(expr, (list, tuple)):
        _expr = []
        for val in expr:
            _expr.append(eval_expr(val, batch=batch, model=model))
        expr = type(expr)(_expr)
    elif isinstance(expr, dict):
        _expr = type(expr)()
        for key, val in expr.items():
            key = eval_expr(key, batch=batch, model=model)
            val = eval_expr(val, batch=batch, model=model)
            _expr.update({key: val})
        expr = _expr
    return expr
