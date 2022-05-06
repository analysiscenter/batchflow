""" !!. """
from textwrap import indent
from torch import nn


class LayerReprMixin:
    """ !!. """
    VERBOSITY_THRESHOLD = 10

    def __repr__(self):
        if hasattr(self, 'verbosity') and self.verbosity < self.VERBOSITY_THRESHOLD:
            return ''

        msg = super().__repr__()
        if getattr(self, 'collapsible', False):
            msg = msg.replace('(\n  (layer): ', ':').replace('\n    ', '\n  ')
            msg = msg.replace('\n  )\n)', '\n)').replace(')\n)', ')')
        return msg

    def prepare_repr(self, verbosity=1, collapsible=True, show_num_parameters=False):
        """ !!. """
        def setter(module):
            setattr(module, 'verbosity', verbosity)
            setattr(module, 'collapsible', collapsible)
            setattr(module, 'show_num_parameters', show_num_parameters)

        def deleter(module):
            try:
                delattr(module, 'verbosity')
                delattr(module, 'collapsible')
                delattr(module, 'show_num_parameters')
            except AttributeError:
                pass

        self.apply(setter)
        msg = repr(self)
        self.apply(deleter)
        return msg

    def repr(self, verbosity=1, collapsible=True, show_num_parameters=False):
        """ !!. """
        print(self.prepare_repr(verbosity=verbosity, collapsible=collapsible, show_num_parameters=show_num_parameters))



class ModuleDictReprMixin(LayerReprMixin):
    """ !!. """
    def __getitem__(self, key):
        if isinstance(key, int):
            key = list(self.keys())[key]
        return super().__getitem__(key)

    def prepare_shape(self, shape, indent=0):
        """ !!. """
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
                            module_short_description += input_shapes + '  ———> '

                            current_line_len = len(module_short_description.splitlines()[-1])
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
