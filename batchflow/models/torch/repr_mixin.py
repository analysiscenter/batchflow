""" Mixins for nn.Modules for better textual visualization. """
from textwrap import indent



class LayerReprMixin:
    """ Adds useful properties and methods for nn.Modules, mainly related to visualization and introspection. """
    VERBOSITY_THRESHOLD = 10

    @property
    def num_frozen_parameters(self):
        return sum(p.numel() for p in self.parameters() if not p.requires_grad)

    @property
    def num_trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def __repr__(self):
        if hasattr(self, 'verbosity') and self.verbosity < self.VERBOSITY_THRESHOLD:
            return ''

        msg = super().__repr__()
        if getattr(self, 'collapsible', False):
            msg = msg.replace('(\n  (layer): ', ':').replace('\n    ', '\n  ')
            msg = msg.replace('\n  )\n)', '\n)').replace(')\n)', ')')
        return msg

    def prepare_repr(self, verbosity=1, collapsible=True, show_num_parameters=False, extra=False):
        """ Set flags on children, call `repr`, delete flags.
        Returns string.
        """
        def set_flags(module):
            setattr(module, 'verbosity', verbosity)
            setattr(module, 'collapsible', collapsible)
            setattr(module, 'show_num_parameters', show_num_parameters)
            setattr(module, 'extra', extra)

        def del_flags(module):
            try:
                delattr(module, 'verbosity')
                delattr(module, 'collapsible')
                delattr(module, 'show_num_parameters')
                delattr(module, 'extra')
            except AttributeError:
                pass

        self.apply(set_flags)
        msg = repr(self)
        self.apply(del_flags)
        return msg

    def repr(self, verbosity=1, collapsible=True, show_num_parameters=False, extra=False):
        """ Set flags on children, call `repr`, delete flags.
        Prints output to stdout.
        """
        print(self.prepare_repr(verbosity=verbosity, collapsible=collapsible,
                                show_num_parameters=show_num_parameters, extra=extra))



class ModuleDictReprMixin(LayerReprMixin):
    """ Mixin to allow `repr` for multiple levels for nn.ModuleDicts.
    Also adds `__getitem__` for convenience.

    Relies on modules having `shapes` dictionary, that for each children stores the information about their
    input and output shapes.

    Depending on `verbosity`, creates string representation for different levels:
        - verbosity 1, modules and their shapes. For example, shapes of `initial_block`, `body` and `head`.
        - verbosity 2, blocks inside modules. For example, shapes of blocks inside Encoder.
        - verbosity 3, blocks inside repeated chains of `:class:~.blocks.Block`. Mainly used for debug purposes.
        - verbosity 4, letters inside each `:class:~.layers.MultiLayer`. For example, each letter inside given block.
        - verbosity 5, PyTorch implementations of each letter inside `:class:~.layers.MultiLayer`.
        - verbosity 6+, default repr of nn.Module.
    For most cases, levels 2 and 4 should be used.

    Additional parameters can be used to show number of parameters inside each level and collapse multilines.
    """
    def __getitem__(self, key):
        if isinstance(key, int):
            key = list(self.keys())[key]
        return super().__getitem__(key)

    def prepare_shape(self, shape, indent=0):
        """ Beautify shape or list of shapes.
        Changes the first dimension (batch) to `?`.
        Makes multiple lines for lists of shapes with provided indentation.
        """
        #pylint: disable=redefined-outer-name
        if isinstance(shape, tuple):
            msg = ', '.join([f'{item:>3}' for item in shape[1:]])
            return f' (?, {msg}) '

        if isinstance(shape, list):
            msg = '[' + self.prepare_shape(shape[0])[1:-1] + ','
            for shape_ in shape[1:]:
                msg += '\n ' + ' '*indent + self.prepare_shape(shape_)[1:-1] + ','
            msg = msg[:-1] + ']'
            return msg
        raise TypeError(f'Should be used on tuple or list of tuples, got {type(shape)} instead.')


    def __repr__(self):
        if hasattr(self, 'verbosity'):
            indent_prefix = '    '

            # Parse verbosity. If equal to max level, set flag
            verbosity = self.verbosity
            if verbosity >= 5:
                verbosity = 4
                detailed_last_level = True
            else:
                detailed_last_level = False

            if len(self.keys()):
                key = list(self.keys())[0]
                input_shapes, output_shapes = None, None

                if (len(self.items()) == 1 and getattr(self, 'collapsible', False)
                    and getattr(self[key], 'VERBOSITY_THRESHOLD', -1) == self.VERBOSITY_THRESHOLD):
                    # Subclasses names can be folded, i.e. `Block:ResBlock(` instead of `Block(\n    ResBlock(`
                    msg = self._get_name() + ':' + repr(self[key])
                    msg = msg.replace(')\n)', ')')

                else:
                    msg = self._get_name() + '(\n'
                    extra_repr = self.extra_repr()
                    if extra_repr:
                        msg += indent(extra_repr, prefix=indent_prefix) + '\n'

                    max_key_length = max(len(key) for key in self.keys())
                    for key, value in self.items():

                        # Short description: module name and description of shapes
                        empty_space = ' ' * (1 + max_key_length - len(key))
                        module_short_description = f'({key}:{empty_space}'

                        if key in self.shapes:
                            input_shapes, output_shapes = self.shapes.get(key)

                            current_line_len = len(module_short_description)
                            input_shapes = self.prepare_shape(input_shapes, indent=current_line_len)
                            module_short_description += input_shapes + ' ⟶ '

                            current_line_len = len(module_short_description.splitlines()[-1]) + 1
                            output_shapes = self.prepare_shape(output_shapes, indent=current_line_len).strip(' ')
                            module_short_description += output_shapes

                        if getattr(self, 'show_num_parameters', False):
                            num_parameters = sum(p.numel() for p in value.parameters() if p.requires_grad)
                            module_short_description += f', #params={num_parameters:,}'
                        module_short_description += ')'

                        # Long description: ~unmodified repr of a module
                        module_long_description = repr(value).strip(' ')

                        # Select appropriate message
                        module_description = ''
                        if verbosity > self.VERBOSITY_THRESHOLD:
                            module_description = f'({key}): ' + module_long_description

                        if verbosity == self.VERBOSITY_THRESHOLD or module_description == f'({key}): ':
                            module_description = module_short_description

                        if self.VERBOSITY_THRESHOLD == 4 and detailed_last_level:
                            module_description = (module_short_description + ':\n' +
                                                  indent(module_long_description, prefix=indent_prefix))

                        msg += indent(module_description, prefix=indent_prefix) + '\n'
                        msg = msg.replace('\n\n', '\n')
                    msg += ')'

                if len(self.items()) == 1 and getattr(self, 'collapsible', False) and 'r0' in msg:
                    msg = msg.replace('(\n    (r0:', '(r0:', 1)
                    msg = msg.replace(')\n)', ')')
            else:
                msg = self._get_name() + '()'
            return msg
        return super().__repr__()


REPR_DOC = '\n'.join(ModuleDictReprMixin.__doc__.split('\n')[3:])
LayerReprMixin.repr.__doc__ += '\n' + REPR_DOC
LayerReprMixin.prepare_repr.__doc__ += '\n' + REPR_DOC
