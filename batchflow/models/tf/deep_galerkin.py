""" Deep Galerkin model for solving partial differential equations. Inspired by
Sirignano J., Spiliopoulos K. "`DGM: A deep learning algorithm for solving partial differential equations
<http://arxiv.org/abs/1708.07469>`_"
"""

import numpy as np
import tensorflow as tf

from . import TFModel


class DeepGalerkin(TFModel):
    r""" Deep Galerkin model for solving partial differential equations (PDEs) of the second order
    with constant coefficients on rectangular domains using neural networks. Inspired by
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
        - Q : callable or const
            if callable, must accept and return tf.Tensor.
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
                Q=5,
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
        config = super().default_config()
        config['common/time_multiplier'] = 'sigmoid'
        config['common/bind_bc_ic'] = True
        return config

    def build_config(self, names=None):
        """ PDE-problem is fetched from 'pde' key in 'self.config',
        and then is passed to 'common' so that all subsequent blocks get it as 'kwargs'.
        """
        pde = self.config.get('pde')
        if pde is None:
            raise ValueError("The PDE-problem is not specified. Use 'pde' config to set up the problem.")

        # get dimensionality
        form = pde.get("form")
        n_dims = len(form.get("d1", form.get("d2", None)))
        self.config.update({'pde/n_dims': n_dims,
                            'initial_block/inputs': 'points',
                            'inputs': dict(points={'shape': (n_dims, )})})

        # default values for domain
        if pde.get('domain') is None:
            self.config.update({'pde/domain': [[0, 1]] * n_dims})

        # default value for rhs
        if pde.get('Q') is None:
            self.config.update({'pde/Q': 0})

        # make sure that initial conditions are callable
        init_cond = pde.get('initial_condition', None)
        if init_cond is not None:
            init_cond = init_cond if isinstance(init_cond, (tuple, list)) else (init_cond, )
            init_cond = [expression if callable(expression) else lambda *args, e=expression:
                         e for expression in init_cond]
            self.config.update({'pde/initial_condition': init_cond})

        # 'common' is updated with PDE-problem
        config = super().build_config(names)
        config['common'].update(self.config['pde'])

        if config.get('output') is not None:
            config['common/track'] = self.parse_output(config['output'], n_dims)
        return config

    def parse_output(self, ops, n_dims):
        """ Parsing engine. """
        forms = []
        if ops is None:
            ops = []
        elif not isinstance(ops, (dict, tuple, list)):
            ops = [ops]
        if not isinstance(ops, dict):
            ops = {'': ops}
        prefix = list(ops.keys())[0]
        _ops = dict()
        _ops[prefix] = list(ops[prefix])

        _map_coords = dict(x=0, y=1, z=2, t=-1)
        for op in _ops[prefix]:
            if isinstance(op, str):
                op = op.replace(" ", "").replace("_", "")
                if op.startswith("d"):
                    # parse order
                    prefix_len = 1
                    try: # for example, d2xy
                        order = int(op[1])
                        prefix_len += 1
                    except ValueError: # for example, dx
                        order = 1

                    if order > 2:
                        raise ValueError("Tracking gradients of order " + order + " is not supported.")

                    # parse variables
                    variables = op[prefix_len:]

                    if len(variables) == 1: # for example, d2x
                        coord_number = _map_coords.get(variables)
                        if coord_number is None: # for example, dr
                            raise ValueError("Cannot parse coordinate number from " + op)
                        if order == 2: # for example, d2x
                            coord_number = [coord_number, coord_number]
                    elif len(variables) == 2: # for example, d2xy
                        try: # for example, d2x0
                            coord_number = int(variables[1:])
                            if order == 2:
                                coord_number = [coord_number, coord_number]
                        except ValueError:
                            coord_number = [_map_coords.get(variables[0]), _map_coords.get(variables[1])]

                    elif len(variables) == 4:
                        try:
                            coord_number = [int(variables[1]), int(variables[3])]
                        except:
                            raise ValueError("Cannot parse coordinate numbers from " + op)

                    if isinstance(coord_number, list):
                        if coord_number[0] is None or coord_number[1] is None:
                            raise ValueError("Cannot parse coordinate numbers from " + op)

                    # make callable to compute required op
                    form = np.zeros((n_dims, )) if order == 1 else np.zeros((n_dims, n_dims))
                    if order == 1:
                        form[coord_number] = 1
                    else:
                        form[coord_number[0], coord_number[1]] = 1
                    form = {"d" + str(order): form}
                    forms.append(form)
        return forms, _ops[prefix]


    def _make_inputs(self, names=None, config=None):
        """ Create needed placeholders. """
        # make sure inputs-placeholder of pde's dimension (x_1, /dots, x_n, t) is created
        placeholders_, tensors_ = super()._make_inputs(names, config)

        n_dims = config.get('pde/n_dims')
        tensors_['points'] = tf.split(tensors_['points'], n_dims, axis=1, name='coordinates')
        tensors_['points'] = tf.concat(tensors_['points'], axis=1)

        # calculate targets-tensor using rhs of pde and created points-tensor
        points = getattr(self, 'inputs').get('points')
        q = config.get('pde/Q')
        if not callable(q):
            if isinstance(q, (float, int)):
                q_val = q
                q = lambda *args: q_val * tf.ones_like(tf.reduce_sum(points, axis=1, keepdims=True))
            else:
                raise ValueError("Cannot parse right-hand-side of the equation")

        self.store_to_attr('targets', q(points))
        return placeholders_, tensors_


    @classmethod
    def _make_form_calculator(cls, form, coordinates, name='_callable'):
        """ Get callable that computes differential form of a tf.Tensor
        with respect to coordinates.
        """
        n_dims = len(coordinates)
        d1_coeffs = np.array(form.get("d1", np.zeros(shape=(n_dims, )))).reshape(-1)
        d2_coeffs = np.array(form.get("d2", np.zeros(shape=(n_dims, n_dims)))).reshape(n_dims, n_dims)

        if (np.all(d1_coeffs == 0) and np.all(d2_coeffs == 0)):
            raise ValueError('Nothing to compute here! Either d1 or d2 must be non-zero')

        def _callable(net):
            """ Compute differential form.
            """
            # derivatives of the first order
            vars = [coordinates[i] for i in np.nonzero(d1_coeffs)[0]]
            result = sum(coeff * d1_ for coeff, d1_ in zip(d1_coeffs[d1_coeffs != 0], tf.gradients(net, vars)))

            # derivatives of the second order
            for i in range(n_dims):
                vars = [coordinates[i] for i in np.nonzero(d2_coeffs[i, :])[0]]
                if len(coordinates) > 0:
                    d1_ = tf.gradients(net, coordinates[i])[0]
                    result += sum(coeff * d2_ for coeff, d2_ in zip(d2_coeffs[i, d2_coeffs[i, :] != 0],
                                                                    tf.gradients(d1_, vars)))
            return result

        setattr(_callable, '__name__', name)
        return _callable

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
        Form an `approximator`-tensor binding the initial value (at $t=0$) of the `network`-tensor to $sin(2 \pi x)$::

            approximator = network * DeepGalerkin._make_time_multiplier('sigmoid', '0')(t) + tf.sin(2 * np.pi * x)

        Bind the initial value to $sin(2 \pi x)$ and the initial rate to $cos(2 \pi x)$::

            approximator = (network * DeepGalerkin._make_time_multiplier('polynomial', '00')(t) +
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


    @classmethod
    def head(cls, inputs, name='head', **kwargs):
        """ Head block of the model. Binds `initial_condition` or `boundary_condition`, if these
        are supplied in the config of the model. Does so by applying one of preset multipliers to
        the network output. Creates a tf.Tensor `approximator` - the final output of the model.

        Implements all features from :meth:`.TFModel.head`. For instance, accepts layout
        for :func:`.conv_block`.
        """
        inputs = super().head(inputs, name, **kwargs)
        if kwargs.get("bind_bc_ic"):
            add_term = 0
            multiplier = 1

            # retrieving variables
            n_dims = kwargs.get('n_dims')
            coordinates = [inputs.graph.get_tensor_by_name(cls.__name__ + '/inputs/coordinates:' + str(i))
                           for i in range(n_dims)]

            domain = kwargs.get("domain")
            lower, upper = [[bounds[i] for bounds in domain] for i in range(2)]

            init_cond = kwargs.get("initial_condition")
            n_dims_xs = n_dims if init_cond is None else n_dims - 1
            xs_spatial = tf.concat(coordinates[:n_dims_xs], axis=1) if n_dims_xs > 0 else None

            # multiplicator for binding boundary conditions
            if n_dims_xs > 0:
                lower_tf, upper_tf = [tf.constant(bounds[:n_dims_xs], shape=(1, n_dims_xs), dtype=tf.float32)
                                      for bounds in (lower, upper)]
                multiplier *= tf.reduce_prod((xs_spatial - lower_tf) * (upper_tf - xs_spatial) /
                                             (upper_tf - lower_tf)**2,
                                             axis=1, name='xs_multiplier', keepdims=True)


            # ingore boundary condition as it is automatically set by initial condition
            if init_cond:
                shifted = coordinates[-1] - tf.constant(lower[-1], shape=(1, 1), dtype=tf.float32)
                time_mode = kwargs.get("time_multiplier")

                add_term += init_cond[0](xs_spatial)
                multiplier *= cls._make_time_multiplier(time_mode, '0' if len(init_cond) == 1 else '00')(shifted)

                # case of second derivative with respect to t in lhs of the equation
                if len(init_cond) > 1:
                    add_term += init_cond[1](xs_spatial) * cls._make_time_multiplier(time_mode, '01')(shifted)

            # apply transformation to inputs
            inputs = add_term + multiplier * inputs

        return tf.identity(inputs, name='approximator')

    def output(self, inputs, predictions=None, ops=None, prefix=None, **kwargs):
        r""" Output block of the model. Computes differential form for lhs of the equation.
        In addition, allows for convenient logging of differentials into output ops. Accepts
        all arguments from original :meth:`.TFModel.output`.

        **Differentials-logging**

        Allows for logging differentials of first and second order w.r.t. any variable. To output
        derivative w.r.t. first coordinate
            $$
                \frac{\partial f}{\partial x_0}
            $$

        simply add::

            config = {
                'output': 'd1x0'
            }

        or, even simpler, as 'x', 'y', 'z' stand for first three coordinates::

            config = {
                'output': 'd1x'
            }

        while the derivative of the second order w.r.t. the last coordinate (time in equations of evolution)
            $$
                \frac{\partial^2 f}{\partial t^2}
            $$

        is output by::

            config = {
                'output': 'd2t'
            }

        """
        self.store_to_attr('approximator', inputs)

        form = kwargs.get("form")
        n_dims = kwargs.get("n_dims")
        coordinates = [inputs.graph.get_tensor_by_name(self.__class__.__name__ + '/inputs/coordinates:' + str(i))
                       for i in range(n_dims)]

        if kwargs.get('track') is not None:
            forms, _ops = kwargs.get('track')
            for i, form in enumerate(forms):
                _compute_op = self._make_form_calculator(form, coordinates, name=_ops[i])
                _ops[i] = _compute_op
        else:
            _ops = ['']

        # differential form from lhs of the equation
        _compute_predictions = self._make_form_calculator(kwargs.get("form"), coordinates, name='predictions')
        return super().output(inputs, _compute_predictions, _ops, prefix, **kwargs)

    def predict(self, fetches=None, feed_dict=None, **kwargs):
        """ Get network-approximation of PDE-solution on a set of points. Overloads :meth:`.TFModel.predict` :
        `approximator`-tensor is now considered to be the main model-output.
        """
        fetches = 'approximator' if fetches is None else fetches
        return super().predict(fetches, feed_dict, **kwargs)
