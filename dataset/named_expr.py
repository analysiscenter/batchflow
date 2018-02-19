""" Contains named expression classes"""
import numpy as np


class _DummyBatch:
    """ A fake batch for static models """
    def __init__(self, pipeline):
        self.pipeline = pipeline


class NamedExpression:
    """ Base class for a named expression """
    def __init__(self, name, copy=False):
        self.name = name
        self.copy = copy

    def get(self, batch=None, pipeline=None, model=None):
        """ Return a value of a named expression """
        if isinstance(self.name, NamedExpression):
            return self.name.get(batch=batch, pipeline=pipeline, model=model)
        return self.name

    def set(self, value, batch=None, pipeline=None, model=None, mode='w'):
        """ Set a value to a named expression """
        value = eval_expr(value, batch=batch, pipeline=pipeline, model=model)
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

    def __repr__(self):
        return type(self).__name__ + '(' + str(self.name) + ')'


class W(NamedExpression):
    """ A wrapper which returns the wrapped named expression without evaluating it

    Examples
    --------
    ::

        N(V('variable'))
        N(B(copy=True))
        N(R('normal', 0, 1, size=B('size')))
    """
    def get(self, batch=None, pipeline=None, model=None):
        """ Return a wrapped named expression """
        _ = batch, pipeline, model
        return self.name

    def assign(self, *args, **kwargs):
        """ Assign a value """
        _ = args, kwargs
        raise NotImplementedError("Assigning a value to a wrapper is not supported")


def eval_expr(expr, batch=None, pipeline=None, model=None):
    """ Evaluate a named expression recursively """
    if batch is None:
        batch = _DummyBatch(pipeline)
    args = dict(batch=batch, pipeline=pipeline, model=model)

    if isinstance(expr, NamedExpression):
        _expr = expr.get(**args)
        if isinstance(_expr, NamedExpression) and not isinstance(expr, W):
            expr = eval_expr(_expr, **args)
        else:
            expr = _expr
    elif isinstance(expr, (list, tuple)):
        _expr = []
        for val in expr:
            _expr.append(eval_expr(val, **args))
        expr = type(expr)(_expr)
    elif isinstance(expr, dict):
        _expr = type(expr)()
        for key, val in expr.items():
            key = eval_expr(key, **args)
            val = eval_expr(val, **args)
            _expr.update({key: val})
        expr = _expr
    return expr


class B(NamedExpression):
    """ Batch component or attribute name

    Notes
    -----
    ``B()`` return the batch itself.

    To avoid unexpected data changes the copy of the batch may be returned, if ``copy=True``.

    Examples
    --------
    ::

        B('size')
        B('images_shape')
        B(copy=True)
    """
    def __init__(self, name=None, copy=True):
        super().__init__(name, copy)

    def get(self, batch=None, pipeline=None, model=None):
        """ Return a value of a batch component """
        name = super().get(batch=batch, pipeline=pipeline, model=model)
        if isinstance(batch, _DummyBatch):
            raise ValueError("Batch expressions are not allowed in static models: B('%s')" % name)
        if name is None:
            return batch.deepcopy() if self.copy else batch
        return getattr(batch, name)

    def assign(self, value, batch=None, pipeline=None, model=None):
        """ Assign a value to a batch component """
        name = super().get(batch=batch, pipeline=pipeline, model=model)
        if name is not None:
            setattr(batch, name, value)


class C(NamedExpression):
    """ A pipeline config option

    Examples
    --------
    ::

        C('model_class')
        C('GPU')
    """
    def get(self, batch=None, pipeline=None, model=None):
        """ Return a value of a pipeline config """
        name = super().get(batch=batch, pipeline=pipeline, model=model)
        pipeline = batch.pipeline if batch is not None else pipeline
        config = pipeline.config or {}

        recursive_names = name.split('/')
        for n in recursive_names:
            config = config.get(n)
        return config

    def assign(self, value, batch=None, pipeline=None, model=None):
        """ Assign a value to a pipeline config """
        name = super().get(batch=batch, pipeline=pipeline, model=model)
        pipeline = batch.pipeline if batch is not None else pipeline
        config = pipeline.config or {}
        config[name] = value


class F(NamedExpression):
    """ A function, method or any other callable that takes a batch or a pipeline and possibly other arguments

    Examples
    --------
    ::

        F(MyBatch.rotate, angle=30)
        F(prepare_data, 115, item=10)
    """
    def __init__(self, name=None, _pass=True, *args, **kwargs):
        super().__init__(name)
        self.args = args
        self.kwargs = kwargs
        self._pass = _pass

    def get(self, batch=None, pipeline=None, model=None):
        """ Return a value from a callable """
        name = super().get(batch=batch, pipeline=pipeline, model=model)
        args = []
        if self._pass:
            if isinstance(batch, _DummyBatch) or batch is None:
                _pipeline = batch.pipeline if batch is not None else pipeline
                args += [_pipeline]
            else:
                args += [batch]
            if model is not None:
                args += [model]
        fargs = eval_expr(self.args, batch=batch, pipeline=pipeline, model=model)
        fkwargs = eval_expr(self.kwargs, batch=batch, pipeline=pipeline, model=model)
        return name(*args, *fargs, **fkwargs)

    def assign(self, *args, **kwargs):
        """ Assign a value by calling a callable """
        _ = args, kwargs
        raise NotImplementedError("Assigning a value with a callable is not supported")

class L(F):
    """ A function, method or any other callable """
    def __init__(self, name=None, *args, **kwargs):
        super().__init__(name, _pass=False, *args, **kwargs)


class V(NamedExpression):
    """ Pipeline variable name

    Examples
    --------
    ::

        V('model_name')
        V('loss_history')
    """
    def get(self, batch=None, pipeline=None, model=None):
        """ Return a value of a pipeline variable """
        name = super().get(batch=batch, pipeline=pipeline, model=model)
        pipeline = batch.pipeline if batch is not None else pipeline
        value = pipeline.get_variable(name)
        return value

    def assign(self, value, batch=None, pipeline=None, model=None):
        """ Assign a value to a pipeline variable """
        name = super().get(batch=batch, pipeline=pipeline, model=model)
        pipeline = batch.pipeline if batch is not None else pipeline
        pipeline.assign_variable(name, value, batch=batch)


class R(NamedExpression):
    """ A random value

    Notes
    -----
    If `size` is needed, it should be specified as a named, not a positional argument.

    Examples
    --------
    ::

        R('normal', 0, 1)
        R('poisson', lam=5.5, seed=42, size=3)
        R(['metro', 'taxi', 'bike'], p=[.6, .1, .3], size=10)
    """
    def __init__(self, name=None, *args, state=None, seed=None, size=None, **kwargs):
        if not (callable(name) or isinstance(name, (str, NamedExpression))):
            args = (name,) + args
            name = 'choice'
        super().__init__(name)
        if isinstance(state, np.random.RandomState):
            self.random_state = state
        else:
            self.random_state = np.random.RandomState(seed)
        self.args = args
        self.kwargs = kwargs
        self.size = size

    def get(self, batch=None, pipeline=None, model=None):
        """ Return a value of a random variable """
        name = super().get(batch=batch, pipeline=pipeline, model=model)
        if callable(name):
            pass
        elif isinstance(name, str) and hasattr(self.random_state, name):
            name = getattr(self.random_state, name)
        else:
            raise TypeError('Random distribution should be a callable or a numpy distribution')
        args = eval_expr(self.args, batch=batch, pipeline=pipeline, model=model)
        if self.size is not None:
            self.kwargs['size'] = self.size
        kwargs = eval_expr(self.kwargs, batch=batch, pipeline=pipeline, model=model)

        return name(*args, **kwargs)

    def assign(self, *args, **kwargs):
        """ Assign a value """
        _ = args, kwargs
        raise NotImplementedError("Assigning a value to a random variable is not supported")

    def __repr__(self):
        return 'R(' + str(self.name) + ', ' + str(self.args) + ', ' + str(self.kwargs) + ')'


class P(W):
    """ A wrapper for parallel actions

    Notes
    -----
    For ``R``-expressions the default ``size`` will be ``B('size')``.

    Examples
    --------
    Each image in the batch will be rotated at its own angle::

        pipeline
            .rotate(angle=P(R('normal', 0, 1)))

    Without ``P`` all images in the batch will be rotated at the same angle,
    as an angle randomized across batches only::

        pipeline
            .rotate(angle=R('normal', 0, 1))

    Generate 10 categorical random samples::

        pipeline
            .calc_route(P(R(['metro', 'taxi', 'bike'], p=[.6, 0.1, 0.3], size=10))

    If a batch size is greater than 10, than an exception will be raised as there is not enough
    values for each parallel invocations of an action.
    """
    def __init__(self, name=None):
        if isinstance(name, R):
            name.size = name.size if name.size is not None else B('size')
        super().__init__(name)

    def get(self, batch=None, pipeline=None, model=None, parallel=False):   # pylint:disable=arguments-differ
        """ Return a wrapped named expression """
        if parallel:
            return self.name.get(batch=batch, pipeline=pipeline, model=model)
        return self
