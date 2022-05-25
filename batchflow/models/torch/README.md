The code and its logic is organized as follows, from top to bottom:
* `TorchModel` class, located in `base.py`:
    * initialized with a config-dictionary
    * responsible for parsing environment (device, CUDA instructions)
    * uses directly supplied model or creates one via `Network` class (see below)
    * provides API for the underlying PyTorch model, e.g. methods for training and predicting
    * stores a lot of information about model training, predicting, loss values, shapes, parameters, etc

* `Network` class, located in `network.py`:
    * initialized with a config-dictionary
    * creates a nn.Module from large logic modules, like `Default, Encoder, Decoder` modules (see below)
    * files `resnet.py`, `unet.py` and others contain examples of configs for defining corresponding networks

* modules are located in `modules` directory:
    * Initialized with an example of input tensor(s) or their shapes + other keyword arguments
    * Almost all of modules are created from blocks (see below)
    * Control and can modify the input and output types of forward passes (tensors or lists of tensors)
    * For example, `EncoderModule` is a sequential of `block⟶downsample⟶block⟶downsample⟶...`, where each block can be further parametrized
    * `DefaultModule` relies on `Block`, but adds parameters to disable at inference, and can also be used for wrapping a ready-to-use nn.Module
    * `EncoderModule` implements logic of gradually decreasing spatial resolution of tensors, and is enough to implement most classification networks (e.g. ResNet, EfficientNet, ConvNext, ViT)
    * `DecoderModule` and `MLPDecoderModule` implement logic of gradually increasing spatial resolution of tensors, and is enough to implement most segmentation networks (e.g. FCN, DeepLab, SegFormer)
    * `modules/loaders.py` contains modules that wrap the process of importing ready-to-use networks from popular libraries

* blocks are located in `blocks` directory:
    * Initialized with an example of input tensor(s) or their shapes + other keyword arguments
    * For actual operations, most of blocks rely on `MultiLayer` (see below)
    * `Block` implements logic of stitching multiple nn.Modules together and repeating them `N` times
    * contains a lot of named blocks, e.g. `ResBlock`, `DenseBlock`, `ConvNextBlock`
    * contains a lot of attention blocks, e.g. `SEBlock`, `SCSEBlock`, `KSAC`
    * almost all of inherited blocks directly subclass `Block` instead of using it as an attribute

* layers are located in `layers` directory:
    * Initialized with an example of input tensor(s) or their shapes + other keyword arguments
    * Essentially, a large collection of thin wrappers around torch.nn layers
    * Use the provided example of input tensor to infer parameters from its shape
    * For example, our `Conv` layer sees the input tensor. Therefore, you need only `out_channels` for layer definition, while `in_channels` can be infered from input tensor
    * `MultiLayer` class defines the logic of ~sequential chaining of multiple layers, based on string layout. For example, `cna` stands for `convolution⟶batchnorm⟶activation`. Other keyword arguments are passed to individual layers.

* Other important points and files are:
    * `ModuleDictReprMixin` from `repr_mixin.py` is used for most of our classes to provide finer control over textual repr of nn.Modules
    * `utils.py` contains utility functions for working with tensors, shapes and parameters
