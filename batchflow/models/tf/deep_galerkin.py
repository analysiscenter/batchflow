""" Deep Galerkin model for solving partial differential equations. Inspired by
Sirignano J., Spiliopoulos K. "`DGM: A deep learning algorithm for solving partial differential equations
<http://arxiv.org/abs/1708.07469>`_"
"""

import numpy as np
import tensorflow as tf

from . import TFModel

def add_binary_magic(cls, operators=('__add__', '__radd__', '__mul__', '__rmul__', '__sub__', '__rsub__',
                                     '__truediv__', '__rtruediv__', '__pow__', '__rpow__')):
    """ Add binary-magic operators to `SyntaxTreeNode`-class.
    """
    for magic_name in operators:
        def magic(self, other, magic_name=magic_name):
            return cls((magic_name, self, other))
        setattr(cls, magic_name, magic)
    return cls

@add_binary_magic
class SyntaxTreeNode(tuple):
    pass

# nullary: variables (arguments fetchers)
MAX_DIM = 10
nullary = {**{'u': lambda *args: args[0], 'x': lambda *args: args[1],
           'y': lambda *args: args[2], 'z': lambda *args: args[3],
           't': lambda *args: args[-1]},
           **{'x' + str(num): lambda *args, num=num: args[num] for num in range(MAX_DIM)}}

# unary: mathematical transformations (`sin` e.g.)
(sin, cos, exp, log, tan, acos,
 asin, atan, sinh, cosh, tanh, asinh, acosh, atanh) = [lambda x, name=name: SyntaxTreeNode((name, x))
                                                       for name in
                                                       ['sin', 'cos', 'exp', 'log', 'tan',
                                                        'acos', 'asin', 'atan', 'sinh', 'cosh',
                                                        'tanh', 'asinh', 'acosh', 'atanh']]

# binary
D = lambda f, x: SyntaxTreeNode(('D', f, x))

def tf_parse(tree):
    """ Parse syntax-tree to tf-callable.
    """
    # constants
    if isinstance(tree, (int, float)):
        return lambda *args: tree

    op = tree[0]
    if len(tree) == 1:
        # nullary
        return nullary[op]
    elif len(tree) == 2:
        # unary
        argument = tree[1]
        return lambda *args: getattr(tf.math, op)(tf_parse(argument)(*args))
    elif len(tree) == 3:
        # binary
        argument_first = tree[1]
        argument_second = tree[2]
        if op == 'D':
            return lambda *args: tf.gradients(tf_parse(argument_first)(*args),
                                              tf_parse(argument_second)(*args))[0]
        else:
            return lambda *args: getattr(tf_parse(argument_first)(*args),
                                         op)(tf_parse(argument_second)(*args))

class DeepGalerkin(TFModel):
    r""" Deep Galerkin model for solving partial differential equations (PDEs) of the second order
    with constant or functional coefficients on rectangular domains using neural networks. Inspired by
    Sirignano J., Spiliopoulos K. "`DGM: A deep learning algorithm for solving partial differential equations
    <http://arxiv.org/abs/1708.07469>`_"

    **Configuration**

    Inherited from :class:`.TFModel`. Supports all config options from  :class:`.TFModel`,
    including the choice of `device`, `session`, `inputs`-configuration, `loss`-function . Also
    allows to set up the network-architecture using options `initial_block`, `body`, `head`. See
    docstring of :class:`.TFModel` for more detail.

    Left-hand-side (lhs), right-hand-side (rhs) and other properties of PDE are defined in `pde`-dict:

    pde : dict
        dictionary of parameters of PDE. Must contain keys
        - form : dict
            may contain keys 'd1' and 'd2', which define the coefficients before differentials
            of first two orders in lhs of the equation.
        - rhs : callable or const
            right-hand-side of the equation. If callable, must accept and return tf.Tensor.
        - domain : list
            defines the rectangular domain of the equation as a sequence of coordinate-wise bounds.
        - bind_bc_ic : bool
            If True, modifies the network-output to bind boundary and initial conditions.
        - initial_condition : callable or const or None or list
            If supplied, defines the initial state of the system as a function of
            spatial coordinates. In that case, PDE is considered to be an evolution equation
            (heat-equation or wave-equation, e.g.). Then, first (n - 1) coordinates are spatial,
            while the last one is the time-variable. If the lhs of PDE contains second-order
            derivative w.r.t time, initial evolution-rate of the system must also be supplied.
            In this case, the arg is a `list` with two callables (constants).
        - time_multiplier : str or callable
            Can be either 'sigmoid', 'polynomial' or callable. Needed if `initial_condition`
            is supplied. Defines the multipliers applied to network for binding initial conditions.
            `sigmoid` works better in problems with asymptotic steady states (heat equation, e.g.).

    `output`-dict allows for logging of differentials of the solution-approximator. Can be used for
    keeping track on the model-training process. See more details here: :meth:`.DeepGalerkin.output`.

    Examples
    --------

        config = dict(
            pde = dict(
                form={'d1': (0, 1), 'd2': ((-1, 0), (0, 0))},
                rhs=5,
                initial_condition=lambda t: tf.sin(2 * np.pi * t),
                bind_bc_ic=True,
                domain=[[0, 1], [0, 3]],
                time_multiplier='sigmoid'),
            output='d1t')

        stands for PDE given by
            \begin{multline}
                \frac{\partial f}{\partial t} - \frac{\partial^2 f}{\partial x^2} = 5, \\
                f(x, 0) = \sin(2 \pi x), \\
                \Omega = [0, 1] \times [0, 3], \\
                f(0, t) = 0 = f(1, t).
            \end{multline}
        while the solution to the equation is searched in the form
            \begin{equation}
                f(x, t) = (\sigma(x / w) - 0.5) * network(x, t) + \sin(x).
            \end{equation}
        We also track
            $$ \frac{\partial f}{\partial t} $$
    """
    @classmethod
    def default_config(cls):
        """ Overloads :meth:`.TFModel.default_config`. """
        config = super().default_config()
        config['ansatz'] = {}
        config['common/time_multiplier'] = 'sigmoid'
        config['common/bind_bc_ic'] = True
        return config

    def build_config(self, names=None):
        """ Overloads :meth:`.TFModel.build_config`.
        PDE-problem is fetched from 'pde' key in 'self.config', and then
        is passed to 'common' so that all of the subsequent blocks get it as 'kwargs'.
        """
        pde = self.config.get('pde')
        if pde is None:
            raise ValueError("The PDE-problem is not specified. Use 'pde' config to set up the problem.")

        # make sure points-tensor is created
        n_dims = pde.get('n_dims')
        self.config.update({'initial_block/inputs': 'points',
                            'inputs': dict(points={'shape': (n_dims, )})})

        # default values for domain
        if pde.get('domain') is None:
            self.config.update({'pde/domain': [[0, 1]] * n_dims})

        # make sure that initial conditions are callable
        init_conds = pde.get('initial_condition', None)
        if init_conds is not None:
            init_conds = init_conds if isinstance(init_conds, (tuple, list)) else [init_conds]
            parsed = []
            for cond in init_conds:
                if callable(cond):
                    # assume it is a function written in dg-language
                    n_dims_xs = n_dims - 1

                    # get syntax-tree and parse it to tf-callable
                    tree = cond(*[SyntaxTreeNode('x' + str(i)) for i in range(n_dims_xs)])
                    parsed.append(tf_parse(tree))
                else:
                    parsed.append(lambda *args, value=cond: value)
            self.config.update({'pde/initial_condition': parsed})

        # make sure that boundary condition is callable
        bound_cond = pde.get('boundary_condition', 0)
        if isinstance(bound_cond, (float, int)):
            bound_cond_value = bound_cond
            self.config.update({'pde/boundary_condition': lambda *args: bound_cond_value})
        elif callable(bound_cond):
            n_dims_xs = n_dims if init_conds is None else n_dims - 1

            # get syntax-tree and parse it to tf-callable
            tree = bound_cond(*[SyntaxTreeNode('x' + str(i)) for i in range(n_dims_xs)])
            self.config.update({'pde/boundary_condition': tf_parse(tree)})
        else:
            raise ValueError("Cannot parse boundary condition of the equation")

        # 'common' is updated with PDE-problem
        config = super().build_config(names)
        config['common'].update(self.config['pde'])

        config = self._make_ops(config)
        return config

    def _make_ops(self, config):
        """ Stores necessary operations in 'config'. """
        # retrieving variables
        ops = config.get('output')
        track = config.get('track')
        n_dims = config['common/n_dims']
        inputs = config.get('initial_block/inputs', config)
        coordinates = [inputs.graph.get_tensor_by_name(self.__class__.__name__ + '/inputs/coordinates:' + str(i))
                       for i in range(n_dims)]

        # ensuring that 'ops' is of the needed type
        if ops is None:
            ops = []
        elif not isinstance(ops, (dict, tuple, list)):
            ops = [ops]
        if not isinstance(ops, dict):
            ops = {'': ops}
        prefix = list(ops.keys())[0]
        _ops = dict()
        _ops[prefix] = list(ops[prefix])

        # forms for tracking
        if track is not None:
            for op in track.keys():
                _compute_op = self._make_form_calculator(track[op], coordinates, name=op)
                _ops[prefix].append(_compute_op)

        # form for output-transformation
        config['predictions'] = self._make_form_calculator(config.get("common/form"), coordinates,
                                                           name='predictions')
        config['output'] = _ops
        return config

    def _make_inputs(self, names=None, config=None):
        """ Create necessary placeholders. """
        placeholders_, tensors_ = super()._make_inputs(names, config)

        # split input so we can access individual variables later
        n_dims = config['pde/n_dims']
        tensors_['points'] = tf.split(tensors_['points'], n_dims, axis=1, name='coordinates')
        tensors_['points'] = tf.concat(tensors_['points'], axis=1)

        # make targets-tensor from zeros
        points = getattr(self, 'inputs').get('points')
        self.store_to_attr('targets', tf.zeros(shape=(tf.shape(points)[0], 1)))
        return placeholders_, tensors_

    @classmethod
    def _make_form_calculator(cls, form, coordinates, name='_callable'):
        """ Get callable that computes differential form of a tf.Tensor
        with respect to coordinates.
        """
        n_dims = len(coordinates)

        # get tree of lhs-differential operator and parse it to tf-callable
        tree = form(SyntaxTreeNode('u'), *[SyntaxTreeNode(('x' + str(i + 1), )) for i in range(n_dims)])
        parsed = tf_parse(tree)

        # `_callable` should be a function of `net`-tensor only
        _callable = lambda net: parsed(net, *coordinates)
        setattr(_callable, '__name__', name)
        return _callable

    def _build(self, config=None):
        """ Overloads :meth:`.TFModel._build`: adds ansatz-block for binding
        boundary and initial conditions.
        """
        inputs = config.pop('initial_block/inputs')
        x = self._add_block('initial_block', config, inputs=inputs)
        x = self._add_block('body', config, inputs=x)
        x = self._add_block('head', config, inputs=x)
        output = self._add_block('ansatz', config, inputs=x)
        self.store_to_attr('solution', output)
        self.output(output, predictions=config['predictions'], ops=config['output'], **config['common'])

    @classmethod
    def ansatz(cls, inputs, **kwargs):
        """ Binds `initial_condition` or `boundary_condition`, if these are supplied in the config
        of the model. Does so by:
        1. Applying one of preset multipliers to the network output
           (effectively zeroing it out on boundaries)
        2. Adding passed condition, so it is satisfied on boundaries
        Creates a tf.Tensor `solution` - the final output of the model.
        """
        if kwargs["bind_bc_ic"]:
            add_term = 0
            multiplier = 1

            # retrieving variables
            n_dims = kwargs['n_dims']
            coordinates = [inputs.graph.get_tensor_by_name(cls.__name__ + '/inputs/coordinates:' + str(i))
                           for i in range(n_dims)]

            domain = kwargs["domain"]
            lower, upper = [[bounds[i] for bounds in domain] for i in range(2)]

            init_cond = kwargs.get("initial_condition")
            bound_cond = kwargs["boundary_condition"]
            n_dims_xs = n_dims if init_cond is None else n_dims - 1
            xs_spatial = coordinates[:n_dims_xs] if n_dims_xs > 0 else []
            xs_spatial_ = tf.concat(xs_spatial, axis=1)

            # multiplicator for binding boundary conditions
            if n_dims_xs > 0:
                lower_tf, upper_tf = [tf.constant(bounds[:n_dims_xs], shape=(1, n_dims_xs), dtype=tf.float32)
                                      for bounds in (lower, upper)]
                multiplier *= tf.reduce_prod((xs_spatial_ - lower_tf) * (upper_tf - xs_spatial_) /
                                             (upper_tf - lower_tf)**2,
                                             axis=1, name='xs_multiplier', keepdims=True)

            # ingore boundary condition as it is automatically set by initial condition
            if init_cond is not None:
                shifted = coordinates[-1] - tf.constant(lower[-1], shape=(1, 1), dtype=tf.float32)
                time_mode = kwargs["time_multiplier"]

                add_term += init_cond[0](*xs_spatial)
                multiplier *= cls._make_time_multiplier(time_mode, '0' if len(init_cond) == 1 else '00')(shifted)

                # multiple initial conditions
                if len(init_cond) > 1:
                    add_term += init_cond[1](*xs_spatial) * cls._make_time_multiplier(time_mode, '01')(shifted)

            # if there are no initial conditions, boundary conditions are used (default value is 0)
            else:
                add_term += bound_cond(*xs_spatial)

            print(inputs.shape, multiplier.shape, add_term)
            # apply transformation to inputs
            inputs = add_term + multiplier * inputs
        return tf.identity(inputs, name='solution')

    @classmethod
    def _make_time_multiplier(cls, family, order=None):
        r""" Produce time multiplier: a callable, applied to an arbitrary function to bind its value
        and, possibly, first order derivataive w.r.t. to time at $t=0$.

        Parameters
        ----------
        family : str or callable
            defines the functional form of the multiplier, can be either `polynomial` or `sigmoid`.
        order : str or None
            sets the properties of the multiplier, can be either `0` or `00` or `01`. '0'
            fixes the value of multiplier as $0$ at $t=0$, while '00' sets both value and derivative to $0$.
            In the same manner, '01' sets the value at $t=0$ to $0$ and the derivative to $1$.

        Returns
        -------
        callable

        Examples
        --------
        Form an `solution`-tensor binding the initial value (at $t=0$) of the `network`-tensor to $sin(2 \pi x)$::

            solution = network * DeepGalerkin._make_time_multiplier('sigmoid', '0')(t) + tf.sin(2 * np.pi * x)

        Bind the initial value to $sin(2 \pi x)$ and the initial rate to $cos(2 \pi x)$::

            solution = (network * DeepGalerkin._make_time_multiplier('polynomial', '00')(t) +
                            tf.sin(2 * np.pi * x) +
                            tf.cos(2 * np.pi * x) * DeepGalerkin._make_time_multiplier('polynomial', '01')(t))
        """
        if family == "sigmoid":
            if order == '0':
                def _callable(shifted_time):
                    log_scale = tf.Variable(0.0, name='time_scale')
                    return tf.sigmoid(shifted_time * tf.exp(log_scale)) - 0.5
            elif order == '00':
                def _callable(shifted_time):
                    log_scale = tf.Variable(0.0, name='time_scale')
                    scale = tf.exp(log_scale)
                    return tf.sigmoid(shifted_time * scale) - tf.sigmoid(shifted_time) * scale - 1 / 2 + scale / 2
            elif order == '01':
                def _callable(shifted_time):
                    log_scale = tf.Variable(0.0, name='time_scale')
                    scale = tf.exp(log_scale)
                    return 4 * tf.sigmoid(shifted_time * scale) / scale - 2 / scale
            else:
                raise ValueError("Order " + str(order) + " is not supported.")

        elif family == "polynomial":
            if order == '0':
                def _callable(shifted_time):
                    log_scale = tf.Variable(0.0, name='time_scale')
                    return shifted_time * tf.exp(log_scale)
            elif order == '00':
                def _callable(shifted_time):
                    return shifted_time ** 2 / 2
            elif order == '01':
                def _callable(shifted_time):
                    return shifted_time
            else:
                raise ValueError("Order " + str(order) + " is not supported.")

        elif callable(family):
            _callable = family
        else:
            raise ValueError("'family' should be either 'sigmoid', 'polynomial' or callable.")

        return _callable

    def predict(self, fetches=None, feed_dict=None, **kwargs):
        """ Get network-approximation of PDE-solution on a set of points. Overloads :meth:`.TFModel.predict` :
        `solution`-tensor is now considered to be the main model-output.
        """
        fetches = 'solution' if fetches is None else fetches
        return super().predict(fetches, feed_dict, **kwargs)
