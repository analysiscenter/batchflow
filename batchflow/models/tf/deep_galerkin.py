""" Deep Galerkin model for solving partial differential equations. """

import numpy as np
import tensorflow as tf

from dataset.models.tf import TFModel


class DeepGalerkin(TFModel):
    """ Deep Galerkin model for solving partial differential equations (PDEs).
    """

    @classmethod
    def initial_block(cls, inputs, name='initial_block', **kwargs):
        """ Initial block of the model.
        """
        # make sure that the rest of the network is computed using separate coordinates
        n_dims = cls.get_shape(inputs)
        inputs = tf.split(inputs, n_dims, name='coordinates')
        inputs = tf.concat(inputs, axis=1)

        return super().initial_block(inputs, name, **kwargs)

    @classmethod
    def _make_form_calculator(cls, form, coordinates, name='_callable'):
        """ Get callable that computes differential form of a tf.Tensor
        with respect to coordinates.
        """
        n_dims = len(coordinates)
        d1_coeffs = form.get("d1", np.zeros(shape=(n_dims, ))).reshape(-1)
        d2_coeffs = form.get("d2", np.zeros(shape=(n_dims, n_dims))).reshape(n_dims, n_dims)

        if (np.all(d1_coeffs == 0) and np.all(d2_coeffs == 0)):
            raise ValueError('Nothing to compute here! Either d1 or d2 must be non-zero')

        def _callable(net):
            """ Compute differential form.
            """
            # derivatives of the first order
            vars = coordinates.reshape(-1)[d1_coeffs != 0]
            result = sum(coeff * d1_ for coeff, d1_ in zip(d1_coeffs[d1_coeffs != 0], tf.gradients(net, vars)))

            # derivatives of the second order
            for i in range(n_dims):
                vars = coordinates.reshape(-1)[d2_coeffs[i, :] != 0]
                if len(coordinates) > 0:
                    d1 = tf.gradients(net, coordinates[i])[0]
                    result += sum(coeff * d2_ for coeff, d2_ in zip(d2_coeffs[i, [d2_coeffs[i, :] != 0]],
                                                                    tf.gradients(d1, vars)))
            return result

        setattr(_callable, '__name__', name)
        return _callable

    @classmethod
    def _make_time_multiplier(mode):
        if mode == "sigmoid":
            def _callable(shifted_time):
                log_scale = tf.Variable(0.0, name='time_scale')
                return tf.sigmoid(shifted_time * tf.exp(log_scale)) - 0.5
        elif mode == "linear":
            def _callable(shifted_time):
                log_scale = tf.Variable(0.0, name='time_scale')
                return shifted_time * tf.exp(log_scale)
        elif callable(mode):
            _callable = mode
        else:
            raise ValueError("mode should be either sigmoid, linear or callable")

        return _callable

    @classmethod
    def head(cls, inputs, name='head', **kwargs):
        inputs = super().head(inputs, name, **kwargs)
        if kwargs.get("bind_bc_ic", True):
            form = kwargs.get("form")
            n_dims = len(form.get("d1", form.get("d2", None)))
            domain = kwargs.get("domain", [[0, 1]] * n_dims)

            # multiplicator for binding boundary conditions
            lower, upper = [[bounds[i] for bounds in domain] for i in range(2)]
            coordinates = [inputs.graph.get_tensor_by_name('coordinates:' + str(i)) for i in range(n_dims)]
            ic = kwargs.get("initial_condition")
            n_dims_xs = n_dims if ic not None else n_dims - 1

            xs = tf.concat(coordinates[:n_dims_xs], axis=1)
            lower_tf, upper_tf = [tf.constant(bounds[:n_dims_xs], shape=(1, n_dims_xs), dtype=tf.float32)
                                  for bounds in (lower, upper)]
            multiplier = tf.reduce_prod((xs - lower_tf) * (upper_tf - xs) / (upper_tf - lower_tf)**2, axis=1)

            # addition term and time-multiplier
            add_term = 0
            if ic is None:
                add_term += kwargs.get("boundary_condition", 0)
            else:
                # ingore boundary condition as it is automatically set by initial condition
                shifted = coordinates[-1:] - tf.constant(lower[-1:], shape=(1, 1), dtype=tf.float32)
                time_mode = kwargs.get("time_multiplier", "sigmoid")
                multiplier *= cls._make_time_multiplier(time_mode)(shifted)
                add_term += ic(coordinates[:n_dims_xs]) if callable(ic) else ic

            # apply transformation to inputs
            inputs = add_term + multiplier * inputs

        return inputs

    @classmethod
    def output(cls, inputs, predictions=None, ops=None, prefix=None, **kwargs):
        """ Output block of the model.

        Computes differential form for lhs of the equation. In addition, allows for convenient
        logging of differentials into output ops.
        """
        form = kwargs.get("form")
        n_dims = len(form.get("d1", form.get("d2", None)))
        coordinates = [inputs.graph.get_tensor_by_name('coordinates:' + str(i)) for i in range(n_dims)]

        # parsing engine for differentials-logging
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
        for i, op in enumerate(_ops[prefix]):
            if isinstance(op, str):
                op = op.replace(" ", "").replace("_", "")
                if op.startswith("d1") or op.startswith("d2"):
                    # parse coordinate number from needed output name
                    order = op[:2]
                    coord_number = _map_coords.get(op[2:])
                    if coord_number is None:
                        prefix_length = 3 if op[2] == "x" else 2
                        try:
                            coord_number = int(op[prefix_length:])
                        except:
                            raise ValueError("Cannot parse coordinate number from " + op)

                    # make callable to compute required op
                    form = np.zeros((n_dims, ))
                    form[coord_number] = 1
                    if order == "d2":
                        form = np.diag(form)
                    form = {order: form}
                    _compute_op = cls._make_form_calculator(form, coordinates, name=op)

                    # write this callable to outputs-dict
                    _ops[prefix][i] = _compute_op

        # differential form from lhs of the equation
        _compute_predictions = cls._make_form_calculator(form, coordinates)
        return super().output(inputs, _compute_predictions, _ops, prefix, **kwargs)
