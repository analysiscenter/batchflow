""" Contains PascalVOC dataset and labels for different tasks """

import os
from os.path import dirname, basename
from io import BytesIO
import tarfile
import tempfile
import urllib
from collections import defaultdict
import logging

import PIL
import tqdm
import numpy as np

from . import ImagesOpenset
from .. import DatasetIndex

logger = logging.getLogger('PascalVOCdataset')

class BasePascal(ImagesOpenset):
    """ The base class for PascalVOC dataset.
    The archive contains 17125 images. Total size 1.9GB.

    Tracks of the PascalVOC challenge:
        1. Classification
        2. Detection
        3. Segmentation
        4. Action Classification Task
        5. Boxless Action Classification
        6. Person Layout

    Notes
    -----
    - Each track contains only the subset of the total images with labels provided.
    """
    SOURCE_URL = 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar'
    sets_path = 'VOCdevkit/VOC2012/ImageSets'
    task = None

    def __init__(self, *args, bar=False, preloaded=None, train_test=True, **kwargs):
        self.bar = tqdm.tqdm(total=2) if  bar else None
        super().__init__(*args, preloaded=preloaded, train_test=train_test, **kwargs)
        if self.bar:
            self.bar.close()

    def download_archive(self, path=None):
        """ Download archive if not yet on the disk """
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

        return localname

    def _extract(self, archive, member):
        data = archive.extractfile(member).read()
        return PIL.Image.open(BytesIO(data))

    def _get_ids(self, archive, part):
        """ Train and test images ids are located in specific for each task folder"""
        part_path = os.path.join(self.sets_path, self.task, part) + '.txt'
        raw_ids = archive.extractfile(part_path)
        list_ids = raw_ids.read().decode().split('\n')
        return list_ids

    def _is_train_image(self, file, train_ids):
        return basename(file.name).split('.')[0] in train_ids \
               and basename(dirname(file.name)) == 'JPEGImages'

    def _is_test_image(self, file, test_ids):
        return basename(file.name).split('.')[0] in test_ids \
               and basename(dirname(file.name)) == 'JPEGImages'


class PascalSegmentation(BasePascal):
    """ Contains 2913 images and masks.

    Notes
    -----
    - Index 0 corresponds to background and index 255 corresponds to 'void' or unlabelled.
    """
    task = 'Segmentation'
    name_classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
                    'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
                    'train', 'tvmonitor', 'void']

    def _is_train_mask(self, member, train_ids):
        return basename(member.name).split('.')[0] in train_ids \
               and basename(dirname(member.name)) == 'SegmentationClass'

    def _is_test_mask(self, member, test_ids):
        return basename(member.name).split('.')[0] in test_ids \
               and basename(dirname(member.name)) == 'SegmentationClass'

    def download(self, path):
        localname = self.download_archive(path)
        with tarfile.open(localname, "r") as archive:
            files_in_archive = archive.getmembers()
            train_ids = self._get_ids(archive, 'train')
            test_ids = self._get_ids(archive, 'val')

            train_images = np.array([self._extract(archive, file) for file in files_in_archive
                                     if self._is_train_image(file, train_ids)], dtype=object)
            train_masks = np.array([self._extract(archive, file) for file in files_in_archive
                                    if self._is_train_mask(file, train_ids)], dtype=object)

            test_images = np.array([self._extract(archive, file) for file in files_in_archive
                                    if self._is_test_image(file, train_ids)], dtype=object)
            test_masks = np.array([self._extract(archive, file) for file in files_in_archive
                                   if self._is_test_mask(file, test_ids)], dtype=object)

            self._train_index = DatasetIndex(np.arange(len(train_images)))
            self._test_index = DatasetIndex(np.arange(len(test_images)))

            return (train_images, train_masks), (test_images, test_masks)


class PascalClassification(BasePascal):
    """ Contains 11540 images and corresponding classes

    Notes
    -----
    - Labels are provided by the vector size 20. '1' stands for the presence atleast of one object from
    coresponding class on the image. '-1' stands for the absence. '0' indicates that the object is presented,
    but can hardly be detected.
     """
    task = 'Main'
    name_classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
                    'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
                    'tvmonitor']

    def _is_class_file(self, file):
        """ Whether file contains labels for specific class. 20 files total. """
        return basename(dirname(file.name)) == self.task and '_trainval.txt' in file.name

    def download(self, path):

        def _name(file):
            """ Return image name without format """
            return basename(file.name).split('.')[0]

        localname = self.download_archive(path)
        with tarfile.open(localname, "r") as archive:
            files_in_archive = archive.getmembers()

            d = defaultdict(list)
            class_files = [file for file in files_in_archive if self._is_class_file(file)]
            for class_file in class_files:
                data = archive.extractfile(class_file).read()
                for row in data.decode().split('\n')[:-1]:
                    key = row.split()[0]
                    value = int(row.split()[1])
                    d[key].append(value)

            train_ids = self._get_ids(archive, 'train')
            test_ids = self._get_ids(archive, 'val')

            train_images = np.array([self._extract(archive, file) for file in files_in_archive \
                                     if self._is_train_image(file, train_ids)], dtype=object)
            train_labels = np.array([d[_name(file)] for file in files_in_archive
                                     if self._is_train_image(file, train_ids)])

            test_images = np.array([self._extract(archive, file) for file in files_in_archive \
                            if self._is_train_image(file, test_ids)], dtype=object)
            test_labels = np.array([d[_name(file)] for file in files_in_archive
                                    if self._is_test_image(file, test_ids)])

            self._train_index = DatasetIndex(np.arange(len(train_images)))
            self._test_index = DatasetIndex(np.arange(len(test_images)))

        return (train_images, train_labels), (test_images, test_labels)
