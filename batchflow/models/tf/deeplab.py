"""
Liang-Chieh Chen, Yukun Zhu, George Papandreou, Florian Schroff, Hartwig Adam.
"`Encoder-Decoder with Atrous Separable
Convolution for Semantic Image Segmentation
<https://arxiv.org/abs/1802.02611>`_"
"""
from . import EncoderDecoder, Xception, MobileNet
from .layers import aspp



class DeepLab(EncoderDecoder):
    """ DeepLab v3+ model archtecture.
    By default, input is compressed with encoder, then worked via Atrous Spatial Pyramid Pooling layer,
    and, finally, restored to original spatial size with simple yet effective decoder module.

    Parameters
    ----------
    inputs : dict
        Dictionary with 'images' (see :meth:`~.TFModel._make_inputs`)

    body : dict
        encoder : dict
        Dictionary with parameters for entry encoding: downsampling of the inputs.
        See :meth:`~.EncoderDecoder.encoder` arguments.

        embedding : dict
        Same as :meth:`~.EncoderDecoder.embedding`. Default is to use ASPP.

        decoder : dict
        Dictionary with parameters for decoding of compressed representation.
        See :meth:`~.EncoderDecoder.decoder` arguments. Default is to use bilinear upsampling.
    """
    @classmethod
    def default_config(cls):
        """ Define model defaults. See :meth: `~.TFModel.default_config` """
        config = super().default_config()
        config['body/encoder'] += dict(base=None, num_stages=None)

        config['body/embedding'] += dict(base=aspp)

        config['body/decoder/upsample'] += dict(layout='b')
        return config



class DeepLabX(DeepLab):
    """ DeepLab v3+ model archtecture with Xception-based encoder.

    Parameters
    ----------
    inputs : dict
        Dictionary with 'images' (see :meth:`~.TFModel._make_inputs`)

    body : dict
        encoder : dict
        Dictionary with parameters for entry encoding: downsampling of the inputs.
        See :meth:`~.EncoderDecoder.encoder` arguments.

            base : TFModel
                Model class. Should implement ``make_encoder`` method. Default is to use Xception

            entry : dict
            Dictionary with parameters for entry encoding: downsampling of the inputs.
            See :meth:`~.Xception` arguments.

            middle : dict
            Dictionary with parameters for middle encoding: thorough processing.
            See :meth:`~.Xception` arguments.

            exit : dict
            Dictionary with parameters for exit encoding: final increase in number of features.
            See :meth:`~.Xception` arguments.

        embedding : dict
        Same as :meth:`~.EncoderDecoder.embedding`. Default is to use ASPP.
    """
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config['body/encoder/base'] = Xception
        config['body/encoder/entry'] = dict(num_stages=None, filters=None, strides=2, combine_op='softsum')
        config['body/encoder/middle'] = dict(num_stages=None, filters=None, strides=1, combine_op='sum')
        config['body/encoder/exit'] = dict(num_stages=None, filters=None, strides=1,
                                           depth_activation=True, combine_op='softsum')
        config['body/encoder/downsample'] += dict(layout=None)
        return config

class DeepLabX8(DeepLabX):
    """ DeepLab with output stride 8 and Xception-based encoder. """
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config['initial_block'] += dict(layout='cnacna', filters=32, kernel_size=3, strides=1)

        config['body/encoder/entry/num_stages'] = 4
        config['body/encoder/entry/filters'] = [[64]*3,
                                                [128]*3,
                                                [256]*3,
                                                [728]*3]
        config['body/encoder/entry/strides'] = [2, 2, 2, 1]

        config['body/encoder/middle/num_stages'] = 8
        config['body/encoder/middle/rate'] = 2 # not sure if actually needed
        config['body/encoder/middle/filters'] = [[728]*3]*8

        config['body/encoder/exit/num_stages'] = 2
        config['body/encoder/exit/rate'] = [2, 4]
        config['body/encoder/exit/filters'] = [[728, 1024, 1024],
                                               [1536, 1536, 2048]]

        config['body/embedding/rates'] = (12, 24, 36)

        config['body/decoder'] += dict(skip=True, num_stages=4, factor=[1, 2, 1, 4])
        return config

class DeepLabX16(DeepLabX):
    """ DeepLab with output stride 16 and Xception-based encoder. """
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config['initial_block'] += dict(layout='cnacna', filters=32, kernel_size=3, strides=1)

        config['body/encoder/entry/num_stages'] = 4
        config['body/encoder/entry/filters'] = [[64]*3,
                                                [128]*3,
                                                [256]*3,
                                                [728]*3]
        config['body/encoder/entry/strides'] = [2, 2, 2, 2]

        config['body/encoder/middle/num_stages'] = 16
        config['body/encoder/middle/rate'] = 2 # not sure if actually needed
        config['body/encoder/middle/filters'] = [[728]*3]*16

        config['body/encoder/exit/num_stages'] = 2
        config['body/encoder/exit/rate'] = [2, 4]
        config['body/encoder/exit/filters'] = [[728, 1024, 1024],
                                               [1536, 1536, 2048]]

        config['body/embedding/rates'] = (6, 12, 18)

        config['body/decoder'] += dict(skip=True, num_stages=4, factor=[1, 4, 1, 4])
        return config


class DeepLabXS(DeepLabX):
    """ DeepLab with output stride 16 and Xception-based encoder.
        Smaller than the others and is used for testing purposes.
    """
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config['initial_block'] += dict(layout='cnacna', filters=32, kernel_size=3, strides=1)

        config['body/encoder/entry/num_stages'] = 4
        config['body/encoder/entry/filters'] = [[6]*3,
                                                [1]*3,
                                                [2]*3,
                                                [7]*3]
        config['body/encoder/entry/strides'] = [2, 2, 2, 2]

        config['body/encoder/middle/num_stages'] = 4
        config['body/encoder/middle/rate'] = 2 # not sure if actually needed
        config['body/encoder/middle/filters'] = [[7]*3]*4

        config['body/encoder/exit/num_stages'] = 2
        config['body/encoder/exit/rate'] = [2, 4]
        config['body/encoder/exit/filters'] = [[7, 10, 10],
                                               [15, 15, 20]]

        config['body/embedding/rates'] = (6, 12, 18)

        config['body/decoder'] += dict(skip=True, num_stages=4, factor=[1, 4, 1, 4])
        return config



class DeepLabM(DeepLab):
    """ DeepLab v3+ model archtecture with MobileNet-based encoder. """
    @classmethod
    def default_config(cls):
        config = super().default_config()
        config['body/encoder/base'] = MobileNet
        return config
