""" Open datasets """
from .base import Openset, ImagesOpenset
from .mnist import MNIST
from .cifar import CIFAR10, CIFAR100
from .imagenette import Imagenette160, Imagenette320, Imagenette, ImageWoof160, ImageWoof320, ImageWoof
from .pascal import PascalClassification, PascalSegmentation
from .coco import COCOSegmentation
