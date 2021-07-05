""" Open datasets """
from .base import Openset, ImagesOpenset
from .mnist import MNIST
from .cifar import CIFAR10, CIFAR100
from .imagenette import (Imagenette160, Imagenette320, Imagenette,
                         Imagenette2_160, Imagenette2_320, Imagenette2,
                         ImageWoof160, ImageWoof320, ImageWoof,
                         ImageWoof2_160, ImageWoof2_320, ImageWoof2)
from .pascal import PascalClassification, PascalSegmentation
from .coco import COCOSegmentation
