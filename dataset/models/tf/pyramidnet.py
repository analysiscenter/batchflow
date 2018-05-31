"""
Dongyoon Han et al. "`Deep Pyramidal Residual Networks
<https://arxiv.org/abs/1610.02915>`_"

"""
from . import ResNet


class PyramidNet(ResNet):
    """ The base PyramidNet model

    **Configuration**

    inputs : dict
        dict with 'images' and 'labels' (see :meth:`._make_inputs`)

    input_block : dict
        parameters for the input block (see :func:`.conv_block`).

    body : dict
        num_blocks : list of int
            number of blocks in each group with the same number of filters.

        block : dict
            widening : int
                an increment of filters number in each block (default=8)

            and other :class:`~.ResNet` block params

    head : dict
        'Vdf' with dropout_rate=.4

    Notes
    -----
    Also see :class:`~.TFModel` and :class:`~.ResNet` configuration.
    """
    @classmethod
    def default_config(cls):
        config = ResNet.default_config()
        config['body/block/widening'] = 8
        config['body/block/zero_pad'] = True
        return config

    @classmethod
    def default_layout(cls, bottleneck, **kwargs):
        """ Define conv block layout """
        return 'nc nac nac n' if bottleneck else 'nc nac n'

    def build_config(self, names=None):
        config = super(ResNet, self).build_config(names)

        if config.get('body/filters') is None:
            w = config['body/block/widening']
            filters = config['input_block/filters']
            config['body/filters'] = []
            for g in config['body/num_blocks']:
                bfilters = [filters +  w * b for b in range(1, g + 1)]
                filters = bfilters[-1]
                config['body/filters'].append(bfilters)

        if config.get('head/units') is None:
            config['head/units'] = self.num_classes('targets')
        if config.get('head/filters') is None:
            config['head/filters'] = self.num_classes('targets')

        return config
