""" Contains class for ResNet """
import tensorflow as tf
from . import TFModel
from .layers import conv_block
from .layers.pooling import global_average_pooling


_INPUT_BLOCK_CONFIG = {'layout': 'cnap', 'filters': 64,
                       'kernel_size': 7,
                       'strides': 2, 'pool_size': 3, 'pool_strides': 2}

_COMMON_ARCH = {'filters': 64, 'strides': 2, 'bottelneck_factor': 4,
                'se_block': 0, 'num_blocks': 4}

_CUSTOM_ARCH = {'ResNet18': {'length_factor': [2, 2, 2, 2], 'bottleneck': False},
                'ResNet34': {'length_factor': [3, 4, 6, 3], 'bottleneck': False},
                'ResNet50': {'length_factor': [3, 4, 6, 3], 'bottleneck': True},
                'ResNet101': {'length_factor': [3, 4, 23, 3], 'bottleneck': True},
                'ResNet152': {'length_factor': [3, 8, 36, 3], 'bottleneck': True}}


class ResNet(TFModel):
    """ ResNet
    https://arxiv.org/abs/1512.03385 (Kaiming He et al, 2015)

    ** Configuration **

    inputs : dict
        dict with keys 'images' and 'masks' (see :meth:`._make_inputs`)
    layout: str
        a sequence of layers
        c - convolution
        n - batch normalization
        a - activation
        p - max pooling
        and others (see :func:`~layers.conv_block.conv_block`).
        Default is 'cna'.


    head_type : str in {'dense', 'conv'}
        type of the classification layers in the end of the model.
        Defaults to 'dense'.



    block_params : dict of dicts: ``{'conv': {}, 'batch_norm': {}}``
        'conv' contains parameters for convolution layers, like initializers,
        regularalizers, etc.
        'batch_norm' contains parameters for batch normalization layers,
        like momentum, intiializers, etc.
    activation : callable
        Defaults to `tf.nn.relu`.
    dropout_rate : float between 0 and 1.
        E.g. `rate=0.1` would drop out 10% of input units.
        Defaults to 0.
    filters : int
        number of filters in the first convolutional block (64 by default).
    num_blocks : int
        number of downsampling/upsampling blocks (4 by default)
    length_factor : int or list of ints
        number of residual blocks with the same number of filters.
        If list it should have the length equal to num_blocks.
        Defaults to 1.
    bottleneck : bool or list of bools
        if True then residual blocks will have 1x1, 3x3, 1x1 convolutional layers.
        If list it should have the same length as the filters.
        Defaults to False.
    bottelneck_factor : int or list of ints
        a multiplicative factor for restored dimension in bottleneck block.
        If list it should have the same length as the filters.
        Defaults to 4, i.e., for block with 64 input filters, there will be 256 filters
        in the output tensor.
    strides : int or list of ints
        strides in the block with expanded filters' dimension.
        If list it should have the same length as the filters.
        Defaults to 2.
    se_block : int or list of ints
        if `se_block != 0`, squeeze and excitation block with
        corresponding squeezing factor will be added.
        If list it should have the same length as the filters.
        Defaults to 0.
        Read more about squeeze and excitation technique: https://arxiv.org/abs/1709.01507.
    input_block_config : dict
        containing keyword arguments that will be passed to the first non-residual
        conv_block (see :func:`~lyers.conv_block.conv_block`).
        Defaults to ``{'layout': 'cnap', 'filters': 64,
                      'kernel_size': 7, 'strides': 2,
                      'pool_size': 3, 'pool_strides': 2})``
        i.e. input_block will consist of 64 7x7 convolutions with stride 2
        and maxpooling of size 3 and pool_stride 2.

    Example
    --------
    Recall that original ResNet has `filters = 64` and `num_blocks=4`.
    Then there are 4 types of ResNet blocks sequences with number of filters equal to
    ``[64, 128, 256, 512]``.
    E.g. the `length_factor = [1, 2, 3, 4]` will make one block with 64 output feature maps,
    2 blocks with 128, 3 blocks with 256, 4 with 512.
    """

    def _build(self):
        names = ['images', 'labels']
        _, inputs = self._make_inputs(names)

        n_classes = self.num_classes('labels')
        data_format = self.data_format('images')

        dim = self.get_from_config('dim', 2)
        if not isinstance(dim, int) or dim < 1 or dim > 3:
            raise ValueError("Number of dimensions should be 1, 2 or 3, but given %d" % dim)

        layout = self.get_from_config('layout', 'cna')
        head_type = self.get_from_config('head_type', 'dense')
        if head_type not in ['dense', 'conv']:
            raise ValueError("Head_type should be dense or conv, but given %s" % head_type)
        block_params = self.get_from_config('block_params', {})
        activation = self.get_from_config('activation', tf.nn.relu)
        dropout_rate = self.get_from_config('dropout_rate', 0.)
        input_block_config = self.get_from_config('input_block_config', _INPUT_BLOCK_CONFIG)

        filters = self.get_from_config('filters', 64)
        num_blocks = self.get_from_config('num_blocks', 4)
        length_factor = self.get_from_config('length_factor', 1)
        strides = self.get_from_config('strides', 2)
        bottleneck = self.get_from_config('bottleneck', False)
        bottelneck_factor = self.get_from_config('bottelneck_factor', 4)
        se_block = self.get_from_config('se_block', 0)

        arch = {'filters': filters, 'length_factor': length_factor,
                'strides': strides, 'bottleneck': bottleneck,
                'bottelneck_factor': bottelneck_factor,
                'se_block': se_block, 'num_blocks': num_blocks}
        arch = self.format_architecture(arch)

        is_training = self.is_training
        kwargs = {'is_training': is_training, 'data_format': data_format,
                  'dropout_rate': dropout_rate, 'activation': activation, **block_params}

        with tf.variable_scope('resnet'):
            net = self.input_block(dim, inputs['images'], input_block_config=input_block_config)
            net = self.body(dim, net, arch, layout, **kwargs)
            net = self.head(dim, net, 'conv', 'Pf', n_classes, units=n_classes, **kwargs)

        predictions = tf.identity(net, name='predictions')
        tf.nn.softmax(net, name='predicted_prob')
        labels_hat = tf.cast(tf.argmax(predictions, axis=1), tf.float32, name='labels_hat')
        labels = tf.cast(tf.argmax(inputs['labels'], axis=1), tf.float32, 'true_labels')
        tf.reduce_mean(tf.cast(tf.equal(labels_hat, labels), \
                                  tf.float32), name='accuracy')


    @staticmethod
    def format_architecture(arch):
        '''
        Formats input arch dictionary to a dictionary where every value is a list
        of length equal to `arch['num_blocks']`.
        Handles exeptions with incorrect lengths and creates default values
        for missing keys.

        Parameters
        ----------

        arch : dict

        Returns
        -------

        arch_list : dict
        '''
        filters = arch.get('filters', 64)
        num_blocks = arch.get('num_blocks', 4)

        if isinstance(filters, int):
            filters = [filters * pow(2, index) for index in range(num_blocks)]

        length_factor = arch.get('length_factor', 1)
        if isinstance(length_factor, int):
            length_factor = [length_factor] * num_blocks

        strides = arch.get('strides', 2)
        if isinstance(strides, int):
            strides = [strides] * num_blocks

        bottleneck = arch.get('bottleneck', False)
        if isinstance(bottleneck, bool):
            bottleneck = [bottleneck] * num_blocks

        bottelneck_factor = arch.get('bottelneck_factor', 4)
        if isinstance(bottelneck_factor, int):
            bottelneck_factor = [bottelneck_factor] * num_blocks

        se_block = arch.get('se_block', 0)
        if isinstance(se_block, int):
            se_block = [se_block] * num_blocks

        arch_list = {'length_factor': length_factor,
                     'strides': strides, 'bottleneck': bottleneck,
                     'bottelneck_factor': bottelneck_factor,
                     'se_block': se_block,
                     'filters': filters}

        for name, lst in arch_list.items():
            if len(lst) != num_blocks:
                raise ValueError("%s should be int or list of length equal to num_blocks"
                                 " but given length is %d" % (name, len(lst)))

        return arch_list


    @staticmethod
    def body(dim, inputs, arch='ResNet18', layout='cna', **kwargs):
        """
        Fully convolutional part of the network
        without classification layers on top

        Parameters
        ----------
        dim : int {1, 2, 3}
            input spatial dimensionionaly
        inputs : tf.Tensor
            input tensor
        arch : str or dict
            if str, 'ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152'.
            A dict should contain following keys: (see :class:`~.ResNet`)
            - filters
            - length_factor
            - strides
            - bottleneck
            - bottelneck_factor
            - se_block
        layout : str
            a sequence of layers
            c - convolution
            n - batch normalization
            a - activation
            p - max pooling
            and others (see :func:`~layers.conv_block.conv_block`).
            Default is 'cna'.
        **kwargs :
            keyword arguments that will be passed to conv_block
            (see :func:`~layers.conv_block.conv_block`).

        Returns
        -------
        tf. tensor
            output tensor
        """
        if isinstance(arch, str):
            try:
                arch = {**_COMMON_ARCH, **_CUSTOM_ARCH[arch.lower()]}
            except KeyError:
                print("if string, arch should be one of the following: resnet18, "
                      "resnet34, resnet50, resnet101, resnet152, but %s was given" % arch)
        arch = ResNet.format_architecture(arch)

        with tf.variable_scope('body'):
            net = inputs
            for index, block_length in enumerate(arch['length_factor']):
                for block_number in range(block_length):
                    net = ResNet.block(dim, net, arch['filters'][index], layout, \
                                       'block-'+str(index), block_number, \
                                       arch['strides'][index], \
                                       arch['bottleneck'][index], \
                                       arch['bottelneck_factor'][index], \
                                       arch['se_block'][index], **kwargs)
            net = tf.identity(net, name='conv_output')
        return net


    @staticmethod
    def input_block(dim, inputs, input_block_config, name='block-'+'input'):
        """
        First non-residual block applied to the input of the network

        Parameters
        ----------
        dim : int {1, 2, 3}
            input spatial dimensionionaly
        inputs : tf.Tensor
            input tensor
        input_block_config : dict
            containing keyword arguments that will be passed to conv_block
            (see :func:`~layers.conv_block.conv_block`).
            If `input_block_config == {}` unchanged inputs will be returned.
        name : str
            scope name

        Returns
        -------
        tf. tensor
            output tensor
        """
        with tf.variable_scope(name):
            if input_block_config:
                net = conv_block(dim, inputs, **input_block_config)
            else:
                return inputs
        return net


    @staticmethod
    def block(dim, inputs, filters, layout, name, block_number, strides,
              bottleneck=False, bottelneck_factor=4, se_block=0, **kwargs):
        """
        Residual block

        Parameters
        ----------

        dim : int {1, 2, 3}
            input spatial dimensionionaly
        inputs : tf.Tensor
            input tensor
        filters : int
            number of filters in the ouput tensor
        layout : str
            a sequence of layers:
                c - convolution
                n - batch normalization
                a - activation
                Defaults to `cna`.
        name : str
            scope name
        block_number : int
            index number of the block in the sequence of residual
            blocks of the same number of filters.
        strides : int
            if block_number != 0 strides is equal to 1 for in residual blocks
            downsampling is performed only paired with an enlargement
            of number of filters' size.
        bottleneck : bool.
            whether to do bottleneck. In original ResNet models bottleneck is used
            for networks with more than 50 layers.
            Defaults to False.
        bottelneck_factor : int.
            enlargement factor for the last layer's number of filters in
            the bottleneck block. Recall that 1x1 convolutions are responsible
            for reducing and then restoring filters' dimension.
            Defaults to 4.
        se_block : int.
            squezing factor for the Squeeze and excitation block.
            Se block will be applyed if se_block > 0.
            Defaults to 0.
        ** kwargs :
            keyword arguments that will be passed to conv_block
            (see :func:`~layers.conv_block.conv_block`).

        Returns
        -------
        tf. tensor
            output tensor
        """

        strides = 1 if block_number != 0 else strides

        name = name + '-' + str(block_number)

        with tf.variable_scope(name):
            if bottleneck:
                output_filters = filters * bottelneck_factor
                x = ResNet.bottleneck_conv(dim, inputs, filters, output_filters, layout, strides,
                                           **kwargs)
            else:
                output_filters = filters
                x = ResNet.original_conv(dim, inputs, filters, layout, strides, **kwargs)

            if se_block > 0:
                x = ResNet.se_block(dim, x, se_block, **kwargs)

            shortcut = inputs
            if block_number == 0:
                shortcut = conv_block(dim, inputs, output_filters, 1, 'c', 'shortcut', \
                                      strides=strides, **kwargs)
            x = tf.add(x, shortcut)
            x = tf.nn.relu(x, name='output')
        return x


    @staticmethod
    def bottleneck_conv(dim, inputs, filters, output_filters, layout, strides, **kwargs):
        """
        A stack of 3 convolutions in residual block with bottleneck

        Parameters
        ----------

        dim : int {1, 2, 3}
            input spatial dimensionionaly
        inputs : tf.Tensor
            input tensor
        filters : int
            number of filters in the first two convolutions
        output_filters : int
            number of filters in the last convolution
        layout: str
            a sequence of layers
            c - convolution
            n - batch normalization
            a - activation
            p - max pooling
            and others (see :func:`~layers.conv_block.conv_block`).
            Default is 'cna'.
        strides : int or list of ints
            stride size in the first convolution.
        **kwargs:
            keyword arguments that will be passed to conv_block
            (see :func:`~layers.conv_block.conv_block`).

        Returns
        -------
        tf. tensor
            output tensor
        """
        x = conv_block(dim, inputs, [filters, filters, output_filters], [1, 3, 1], \
                       layout*3, name='bottleneck_conv', strides=[strides, 1, 1], **kwargs)
        return x


    @staticmethod
    def original_conv(dim, inputs, filters, layout, strides, **kwargs):
        """
        A stack of 2 convolutions in residual block

        Parameters
        ----------

        dim : int {1, 2, 3}
            input spatial dimensionionaly
        inputs : tf.Tensor
            input tensor
        filters : int
            number of filters in convolutions
        layout : str
            a sequence of layers
            c - convolution
            n - batch normalization
            a - activation
            p - max pooling
            and others (see :func:`~layers.conv_block.conv_block`).
            Default is 'cna'.
        strides : int or list of ints
            stride size in the first convolution.
        **kwargs :
            keyword arguments that will be passed to conv_block
            (see :func:`~layers.conv_block.conv_block`).

        Returns
        -------
        tf. tensor
            output tensor
        """
        x = conv_block(dim, inputs, filters=[filters, filters], kernel_size=[3, 3], \
                       layout=layout+'d'+layout, strides=[strides, 1], **kwargs)
        return x


    @staticmethod
    def se_block(dim, inputs, se_block, **kwargs):
        """
        Squeeze and excitation block

        Parameters
        ----------

        dim : int {1, 2, 3}
            input spatial dimensionionaly
        inputs : tf.Tensor
            input tensor
        se_block : int
            if `se_block != 0`, squeeze and excitation block with
            corresponding squeezing factor will be added.
            If list it should have the same length as the filters.
            Defaults to 0.
            Read more about squeeze and excitation technique: https://arxiv.org/abs/1709.01507.
        **kwargs :
            keyword arguments that will be passed to conv_block
            (see :func:`~layers.conv_block.conv_block`).

        Returns
        -------
        tf. tensor
            output tensor
        """

        data_format = kwargs['data_format']
        full = global_average_pooling(dim=dim, inputs=inputs, data_format=data_format)
        if data_format == 'channels_last':
            original_filters = inputs.get_shape().as_list()[-1]
            shape = [-1] + [1] * dim + [original_filters]
        else:
            original_filters = inputs.get_shape().as_list()[1]
            shape = [original_filters] + [-1] + [1] * dim
        full = tf.reshape(full, shape)
        full = tf.layers.dense(full, int(original_filters/se_block), activation=tf.nn.relu, \
                               kernel_initializer=tf.contrib.layers.xavier_initializer(), \
                               name='first_dense_se_block')
        full = tf.layers.dense(full, original_filters, activation=tf.nn.sigmoid, \
                               kernel_initializer=tf.contrib.layers.xavier_initializer(), \
                               name='second_dense_se_block')
        return inputs * full


class ResNet152(ResNet):
    """ An original ResNet-101 architecture for ImageNet
    """
    def _build(self):
        self.config['length_factor'] = [3, 8, 36, 3]
        self.config['bottelneck'] = True
        super()._build()


class ResNet101(ResNet):
    """ An original ResNet-101 architecture for ImageNet
    """
    def _build(self):
        self.config['length_factor'] = [3, 4, 23, 3]
        self.config['bottelneck'] = True
        super()._build()


class ResNet50(ResNet):
    """ An original ResNet-50 architecture for ImageNet
    """
    def _build(self):
        self.config['length_factor'] = [3, 4, 6, 3]
        self.config['bottelneck'] = True
        super()._build()


class ResNet34(ResNet):
    """ An original ResNet-34 architecture for ImageNet
    """
    def _build(self):
        self.config['length_factor'] = [3, 4, 6, 3]
        super()._build()


class ResNet18(ResNet):
    """ An original ResNet-18 architecture for ImageNet
    """
    def _build(self, *args, **kwargs):
        self.config['length_factor'] = [2, 2, 2, 2]
        super()._build()
