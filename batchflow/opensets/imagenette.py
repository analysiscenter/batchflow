""" Contains Imagenette and Imagewoof datasets """

import os
from os.path import dirname, basename
import tempfile
import logging
import urllib.request
import tarfile
from io import BytesIO

import PIL
import tqdm
import numpy as np

from . import ImagesOpenset


logger = logging.getLogger('SmallImagenet')


class Imagenette(ImagesOpenset):
    """ Imagenette dataset.
    Contains 12894 train and 500 test images. Total size 1.4GB.

    Notes
    -----
    - Datasets contain both grayscale and colored images, ratio ~ 1:100
      Argument `drop_grayscale` controls whether grayscale images should be dropped.

    """
    SOURCE_URL = 'https://s3.amazonaws.com/fast-ai-imageclas/imagenette.tgz'
    CLASSES = ['tench', 'English springer', 'cassette player', 'chain saw', 'church',
               'French horn', 'garbage truck', 'gas pump', 'golf ball', 'parachute']
    num_classes = 10

    def __init__(self, *args, drop_grayscale=True, bar=False, preloaded=None, train_test=True, **kwargs):
        self.bar = tqdm.tqdm(total=2) if bar else None
        self.drop_grayscale = drop_grayscale
        super().__init__(*args, preloaded=preloaded, train_test=train_test, **kwargs)
        if self.bar:
            self.bar.close()

    def download(self, path=None):
        """ Load data from website and extract it into numpy arrays """

        def _image_class(filepath):
            """ Image's class is determined by the parent folder of the image """
            return basename(dirname(filepath))

        def _is_train(filepath):
            """ Whether image belongs to train or val parts can be determined by
            the level 2 parent folder of the image
            """
            return basename(dirname(dirname(filepath))) == 'train'

        def _extract(archive, member):
            data = archive.extractfile(member).read()
            return PIL.Image.open(BytesIO(data))

        def _is_file_rgb(archive, member):
            """ Check whether archive member is a file.
            In case `drop_grayscale` set to `True` it verifies that the member is the RGB mode image as well.
            """
            if (member.name.find('csv') != -1) or (member.name.find('.DS_Store') != -1):
                return False
            if not self.drop_grayscale:
                return member.isfile()

            return member.isfile() and _extract(archive, member).mode == 'RGB'

        def _gather_extracted(archive, files):
            images = self.create_array([_extract(archive, file) for file in files])
            labels = np.array([_image_class(file.name) for file in files])
            _, labels_encoded = np.unique(labels, return_inverse=True)
            return images, labels_encoded

        if path is None:
            path = tempfile.gettempdir()
        filename = os.path.basename(self.SOURCE_URL)
        localname = os.path.join(path, filename)

        if not os.path.isfile(localname):
            logger.info("Downloading %s", filename)
            urllib.request.urlretrieve(self.SOURCE_URL, localname)
            logger.info("Downloaded %s", filename)
            if self.bar:
                self.bar.update(1)

        logger.info("Extracting...")
        with tarfile.open(localname, "r:gz") as archive:
            files_in_archive = archive.getmembers()

            train_files = [file for file in files_in_archive if _is_file_rgb(archive, file) and _is_train(file.name)]
            train_data = _gather_extracted(archive, train_files)

            test_files = [file for file in files_in_archive if _is_file_rgb(archive, file) and not _is_train(file.name)]
            test_data = _gather_extracted(archive, test_files)

        logger.info("Extracted")
        if self.bar:
            self.bar.update(1)

        images = np.concatenate([train_data[0], test_data[0]])
        labels = np.concatenate([train_data[1], test_data[1]])
        preloaded = images, labels

        train_len, test_len = len(train_data[0]), len(test_data[0])
        index, train_index, test_index = self._infer_train_test_index(train_len, test_len)

        return preloaded, index, train_index, test_index


class Imagenette2(Imagenette):
    """ Imagenette dataset.
    Contains 9296 train and 3856 test images. Total size 1.5GB.
    """
    SOURCE_URL = 'https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz'


class Imagenette320(Imagenette):
    """ The '320px' version of Imagenette.
    The shortest size resized to that size with their aspect ratio maintained.
    Contains 12894 train and 500 test images. Total size 325MB.
    """
    SOURCE_URL = 'https://s3.amazonaws.com/fast-ai-imageclas/imagenette-320.tgz'

class Imagenette2_320(Imagenette):
    """ The '320px' version of Imagenette.
    The shortest size resized to that size with their aspect ratio maintained.
    Contains 9296 train and 3856 test images. Total size 326MB.
    """
    SOURCE_URL = 'https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz'


class Imagenette160(Imagenette):
    """ The '160px' version of Imagenette.
    The shortest size resized to that size with their aspect ratio maintained.
    Contains 12894 train and 500 test images. Total size 98MB.
    """
    SOURCE_URL = 'https://s3.amazonaws.com/fast-ai-imageclas/imagenette-160.tgz'

class Imagenette2_160(Imagenette):
    """ The '160px' version of Imagenette.
    The shortest size resized to that size with their aspect ratio maintained.
    Contains 9296 train and 3856 test images. Total size 94MB.
    """
    SOURCE_URL = 'https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz'


class ImageWoof(Imagenette):
    """ Imagewoof dataset. See the https://github.com/fastai/imagenette for details.
    Contains 12454 train and 500 test images. Total size 1.3GB.
    """
    SOURCE_URL = 'https://s3.amazonaws.com/fast-ai-imageclas/imagewoof.tgz'
    CLASSES = ['Shih-Tzu', 'Rhodesian ridgeback', 'Beagle', 'English foxhound', 'Border terrier',
               'Australian terrier', 'Golden retriever', 'Old English sheepdog', 'Samoyed', 'Dingo']


class ImageWoof2(ImageWoof):
    """ Imagewoof dataset. See the https://github.com/fastai/imagenette for details.
    Contains 8943 train and 3890 test images. Total size 1.25GB.
    """
    SOURCE_URL = 'https://s3.amazonaws.com/fast-ai-imageclas/imagewoof2.tgz'


class ImageWoof320(ImageWoof):
    """ The '320px' version of Imagewoof.
    The shortest size resized to that size with their aspect ratio maintained.
    Contains 12454 train and 500 test images. Total size 313MB.
    """
    SOURCE_URL = 'https://s3.amazonaws.com/fast-ai-imageclas/imagewoof-320.tgz'


class ImageWoof2_320(ImageWoof):
    """ The '320px' version of Imagewoof.
    The shortest size resized to that size with their aspect ratio maintained.
    Contains 8943 train and 3890 test images. Total size 313MB.
    """
    SOURCE_URL = 'https://s3.amazonaws.com/fast-ai-imageclas/imagewoof2-320.tgz'


class ImageWoof160(ImageWoof):
    """ The '160px' version of Imagewoof.
    The shortest size resized to that size with their aspect ratio maintained.
    Contains 12454 train and 500 test images. Total size 88MB
    """
    SOURCE_URL = 'https://s3.amazonaws.com/fast-ai-imageclas/imagewoof-160.tgz'

class ImageWoof2_160(ImageWoof):
    """ The '160px' version of Imagewoof.
    The shortest size resized to that size with their aspect ratio maintained.
    Contains 8943 train and 3890 test images. Total size 88MB.
    """
    SOURCE_URL = 'https://s3.amazonaws.com/fast-ai-imageclas/imagewoof2-160.tgz'
