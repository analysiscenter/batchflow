""" EfficientNet family of NNs

Mingxing Tan, Quoc V. Le "`EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks
<https://arxiv.org/abs/1905.11946>`_"
"""
#pylint: disable=too-many-ancestors
from math import ceil

import torch

from .base import TorchModel
from .blocks import MBConvBlock


def swish(x):
    return x * torch.sigmoid(x)


class EfficientNetB0(TorchModel):
    """
    EfficientNetB0

    Parameters
    ----------
    encoder : dict, optional
        Parameters for model's body

        num_stages : int, optional
            Number of blocks

        order : str, sequence of str, optional
            Determines order of applying layers
            Since no skips are used in EfficientNets this should be always `['block']`

        blocks : dict, optional
            Parameters for pre-processing blocks.
            If parameter's value is a list, each element correspond to different stage

            base_block : callable or list of callable
                Tensor processing function. Default is :class:`~MBConvBlock`

            scalable : bool or list of bool
                Indicates whether the block can be scaled

            other : optional
                Parameters for :class:`~MBConvBlock`

    common : dict, optional
        Common parameters
        width_factor, depth_factor : float, optional
            Scaling factors to control network resizing width-wise and depth-wise
    """

    resolution = 224  # image resolution used in original paper

    @classmethod
    def default_config(cls):
        """ Define model defaults. """
        config = super().default_config()

        config.update({
            'initial_block': {
                'scalable': True,
                'layout': 'cna',
                'kernel_size': 3,
                'stride': 2,
                'channels': 32,
            },
            'body': {
                'scalable': True,
                'type': 'encoder',
                'output_type': 'tensor',
                'num_stages': 7,
                'order': ['block'],
                'blocks': {
                    'scalable': True,
                    'base_block': MBConvBlock,
                    'n_reps': [1, 2, 2, 3, 3, 4, 1],
                    'kernel_size': [3, 3, 5, 3, 5, 5, 3],
                    'stride': [1, 2, 2, 2, 1, 2, 1],
                    'channels': [16, 24, 40, 80, 112, 192, 320],
                    'expand_ratio': [1, 6, 6, 6, 6, 6, 6],
                    'attention': 'se',
                }
            },
            'head': {
                'scalable': True,
                'layout': 'cna V df',
                'kernel_size': 1,
                'stride': 1,
                'channels': 1280,
                'features': None,
                'dropout_rate': 0.2
            },

            'common': {
                'activation': swish,
                'width_factor': 1.0,
                'depth_factor': 1.0,
            },
            'loss': 'ce',
        })
        return config

    def update_config(self):
        """ Define model's architecture configuration. """
        super().update_config()
        config = self.config

        w_factor = config.get('common/width_factor')
        d_factor = config.get('common/depth_factor')

        for path in ('initial_block/', 'body/blocks/', 'head/'):
            scalable = config.get(path + 'scalable')
            if scalable:
                for param, factor in [('channels', w_factor), ('n_reps', d_factor)]:
                    if factor != 1:
                        val = config.get(path + param)
                        if val:
                            if isinstance(val, int):
                                val = max(1, ceil(val * factor))
                            elif isinstance(val, list):
                                val = [max(1, ceil(v * factor)) for v in val]
                            else:
                                raise ValueError(f"`{param}` should be int or list, got {type(val)} instead!")
                            config[path + param] = val

        return config


class EfficientNetB1(EfficientNetB0):
    """ EfficientNetB1. """
    resolution = 240

    @classmethod
    def default_config(cls):
        config = super().default_config()
        config.update({
            'head/dropout_rate': 0.2,
            'common': {
                'width_factor': 1.0,
                'depth_factor': 1.1,
            }
        })
        return config


class EfficientNetB2(EfficientNetB0):
    """ EfficientNetB2. """
    resolution = 260

    @classmethod
    def default_config(cls):
        config = super().default_config()
        config.update({
            'head/dropout_rate': 0.3,
            'common': {
                'width_factor': 1.1,
                'depth_factor': 1.2,
            }
        })
        return config


class EfficientNetB3(EfficientNetB0):
    """ EfficientNetB3. """
    resolution = 300

    @classmethod
    def default_config(cls):
        config = super().default_config()
        config.update({
            'head/dropout_rate': 0.3,
            'common': {
                'width_factor': 1.2,
                'depth_factor': 1.4,
            }
        })
        return config


class EfficientNetB4(EfficientNetB0):
    """ EfficientNetB4. """
    resolution = 380

    @classmethod
    def default_config(cls):
        config = super().default_config()
        config.update({
            'head/dropout_rate': 0.4,
            'common': {
                'width_factor': 1.4,
                'depth_factor': 1.8,
            }
        })
        return config


class EfficientNetB5(EfficientNetB0):
    """ EfficientNetB5. """
    resolution = 456

    @classmethod
    def default_config(cls):
        config = super().default_config()
        config.update({
            'head/dropout_rate': 0.4,
            'common': {
                'width_factor': 1.6,
                'depth_factor': 2.2,
            }
        })
        return config


class EfficientNetB6(EfficientNetB0):
    """ EfficientNetB6. """
    resolution = 528

    @classmethod
    def default_config(cls):
        config = super().default_config()
        config.update({
            'head/dropout_rate': 0.5,
            'common': {
                'width_factor': 1.8,
                'depth_factor': 2.6,
            }
        })
        return config


class EfficientNetB7(EfficientNetB0):
    """ EfficientNetB7. """
    resolution = 600

    @classmethod
    def default_config(cls):
        config = super().default_config()
        config.update({
            'head/dropout_rate': 0.5,
            'common': {
                'width_factor': 2.0,
                'depth_factor': 3.1,
            }
        })
        return config
