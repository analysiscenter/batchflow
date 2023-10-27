""" Contains ADE dataset for semantic segmentation tasks """

import os
from io import BytesIO
from zipfile import ZipFile
import tempfile

from PIL import Image
import tqdm
import requests

from . import ImagesOpenset


class ADESegmentation(ImagesOpenset):
    """ Contains 20210 images and masks for training and 2000 for testing.

    Notes
    -----
    Class 0 corresponds to background.
    """

    SOURCE_URL = 'http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip'
    SETS_PATH = 'ADEChallengeData2016'

    def __init__(self, *args, unpack=False, preloaded=None, train_test=True, **kwargs):
        self.localname = None
        super().__init__(*args, preloaded=preloaded, train_test=train_test, **kwargs)
        if unpack:
            with ZipFile(self.localname) as archive:
                archive.extractall(os.path.dirname(self.localname))

    def download_archive(self, path=None):
        """ Download archive"""
        if path is None:
            path = tempfile.gettempdir()
        filename = os.path.basename(self.SOURCE_URL)
        localname = os.path.join(path, filename)
        self.localname = localname

        if not os.path.isfile(localname):
            r = requests.get(self.SOURCE_URL, stream=True, timeout=10)
            file_size = int(r.headers['Content-Length'])
            chunk = 1
            chunk_size = 1024
            num_bars = int(file_size / chunk_size)
            with open(localname, 'wb') as file:
                for chunk in tqdm.tqdm(
                    r.iter_content(chunk_size=chunk_size),
                    total=num_bars,
                    unit='KB',
                    desc=filename,
                    leave=True
                ):
                    file.write(chunk)

    def _name(self, path):
        """
        Return file name without format

        Parameters
        ----------
        path: str
            path from which you want to extract filename

        Returns
        -------
        str
            file name
        """
        return os.path.basename(path).split('.')[0]

    def _zip_listdir(self, archive, target_dir):
        """
        Analog of os.listdir() but for zipfile

        Parameters
        ----------
        archive: ZipFile object
            .zip archive
        target_dir: str
            directory in which you want to run os.listdir

        Returns
        -------
        List[str]
            list of paths inside target_dir
        """
        paths = archive.namelist()
        target_dir = target_dir if target_dir.endswith("/") else target_dir + "/"
        target_dir = "" if target_dir == '/' else target_dir
        result = [
            path for path in paths if path.startswith(target_dir) and len(path) != len(target_dir)
        ]
        return result

    def _extract_names(self, archive, mode):
        """
        Train and test images names are located in specific for each task folder

        Parameters
        ----------
        archive: ZipFile object
            .zip archive
        mode: str
            can be either "training" or "validation"(test set)

        Returns
        -------
        List[str]
            list of file names
        """
        assert mode in ['training', 'validation']
        target_dir = os.path.join(self.SETS_PATH, 'images', mode)
        filepaths = self._zip_listdir(archive=archive, target_dir=target_dir)
        filenames = [self._name(filepath) for filepath in filepaths]
        return filenames

    def _image_path(self, name, mode):
        """ Return the path to the .jpg image in the archive by its name """
        assert mode in ['training', 'validation']
        return os.path.join(self.SETS_PATH, 'images', mode, name + '.jpg')

    def _mask_path(self, name, mode):
        """ Return the path in the archive to the mask which is .png image by its name and mode"""
        assert mode in ['training', 'validation']
        return os.path.join(self.SETS_PATH, 'annotations', mode, name + '.png')

    def _extract_sample(self, archive, name, mode):
        """
        Return image and mask PIL.Image objects from archive based on its name and mode

        Parameters
        ----------
        archive: ZipFile object
            .zip archive
        name: str
            file name
        mode: str
            can be either "training" or "validation"(test set)

        Returns
        -------
        Tuple(PIL.Image, PIL.Image)
            tuple of image and corresponding mask

        Notes
        -----
        Images that are grayscale are casted to RGB
        """
        image_filepath = self._image_path(name=name, mode=mode)
        mask_filepath = self._mask_path(name=name, mode=mode)
        image_data = archive.read(image_filepath)
        mask_data = archive.read(mask_filepath)

        image = Image.open(BytesIO(image_data))
        image = image.convert('RGB') if (image.mode != 'RGB') else image
        mask = Image.open(BytesIO(mask_data))
        return (image, mask)

    def download(self, path):
        """ Download a dataset from the source web-site """
        self.download_archive(path)
        with ZipFile(self.localname) as archive:
            train_names = self._extract_names(archive=archive, mode='training')
            test_names = self._extract_names(archive=archive, mode='validation')

            train_samples = [self._extract_sample(archive, name=name, mode='training') \
                             for name in train_names]
            test_samples = [self._extract_sample(archive, name=name, mode='validation') \
                            for name in test_names]
            train_images, train_masks = map(list, zip(*train_samples))
            test_images, test_masks = map(list, zip(*test_samples))

            images = self.create_array(train_images + test_images)
            masks = self.create_array(train_masks + test_masks)

            preloaded = images, masks

            index, train_index, test_index = self._infer_train_test_index(
                train_len=len(train_names),
                test_len=len(test_names)
            )

            return preloaded, index, train_index, test_index
