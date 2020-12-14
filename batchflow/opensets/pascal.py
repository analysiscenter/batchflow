""" Contains PascalVOC dataset and labels for different tasks """

import os
from os.path import dirname, basename
from io import BytesIO
import tarfile
import tempfile
from collections import defaultdict

import PIL
import tqdm
import numpy as np
import requests

from . import ImagesOpenset


class BasePascal(ImagesOpenset):
    """ The base class for PascalVOC dataset.
    The archive contains 17125 images. Total size 1.9GB.
    You can unpack the archive to the directory its been downloaded by specifing `unpack` flag.

    Tracks of the PascalVOC challenge:
        1. Classification
        2. Detection
        3. Segmentation
        4. Action Classification Task
        5. Boxless Action Classification
        6. Person Layout

    Notes
    -----
    Each track contains only the subset of the total images with labels provided.
    """
    SOURCE_URL = 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar'
    SETS_PATH = 'VOCdevkit/VOC2012/ImageSets'
    task = None

    def __init__(self, *args, unpack=False, preloaded=None, train_test=True, **kwargs):
        self.localname = None
        super().__init__(*args, preloaded=preloaded, train_test=train_test, **kwargs)
        if unpack:
            with tarfile.open((self.localname), "r") as archive:
                archive.extractall(dirname(self.localname))

    def download_archive(self, path=None):
        """ Download archive"""
        if path is None:
            path = tempfile.gettempdir()
        filename = os.path.basename(self.SOURCE_URL)
        localname = os.path.join(path, filename)
        self.localname = localname

        if not os.path.isfile(localname):
            r = requests.get(self.SOURCE_URL, stream=True)
            file_size = int(r.headers['Content-Length'])
            chunk = 1
            chunk_size = 1024
            num_bars = int(file_size / chunk_size)
            with open(localname, 'wb') as f:
                for chunk in tqdm.tqdm(r.iter_content(chunk_size=chunk_size), total=num_bars, unit='KB',
                                       desc=filename, leave=True):
                    f.write(chunk)

    def _name(self, path):
        """ Return file name without format """
        return basename(path).split('.')[0]

    def _image_path(self, name):
        """ Return the path to the .jpg image in the archive by its name """
        return os.path.join(dirname(self.SETS_PATH), 'JPEGImages', name + '.jpg')

    def _extract_image(self, archive, file):
        data = archive.extractfile(file).read()
        return PIL.Image.open(BytesIO(data))

    def _extract_ids(self, archive, part):
        """ Train and test images ids are located in specific for each task folder"""
        part_path = os.path.join(self.SETS_PATH, self.task, part) + '.txt'
        raw_ids = archive.extractfile(part_path)
        list_ids = raw_ids.read().decode().split('\n')
        return list_ids[:-1]

class PascalSegmentation(BasePascal):
    """ Contains 2913 images and masks.

    Notes
    -----
    Class 0 corresponds to background and class 255 corresponds to 'void' or unlabelled.
    """
    task = 'Segmentation'
    classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'dining table', 'dog', 'horse', 'motorbike', 'person', 'potted plant', 'sheep', 'sofa',
               'train', 'TV/monitor']

    def _mask_path(self, name):
        """ Return the path in the archive to the mask which is .png image by its name"""
        return os.path.join(dirname(self.SETS_PATH), 'SegmentationClass', name + '.png')

    def download(self, path):
        """ Download a dataset from the source web-site """
        self.download_archive(path)
        with tarfile.open(self.localname, "r") as archive:
            train_ids = self._extract_ids(archive, 'train')
            test_ids = self._extract_ids(archive, 'val')

            images = np.array([self._extract_image(archive, self._image_path(name)) \
                               for name in  [*train_ids, *test_ids]], dtype=object)
            masks = np.array([self._extract_image(archive, self._mask_path(name)) \
                              for name in [*train_ids, *test_ids]], dtype=object)
            preloaded = images, masks

            train_len, test_len = len(train_ids), len(test_ids)
            index, train_index, test_index = self._infer_train_test_index(train_len, test_len)

            return preloaded, index, train_index, test_index


class PascalClassification(BasePascal):
    """ Contains 11540 images and corresponding classes

    Notes
    -----
    - Labels are represented by the vector of size 20. '1' stands for the presence of at least one object from
    coresponding class on the image. '-1' stands for the absence. '0' indicates that the object is presented,
    but can hardly be detected.

    - You can control zeros among targets via `process_zero` keyword argument:
    Setting `process_zero=0` leaves the targets as it is.
    Setting `process_zero=1` or any other positive number replaces `0` with `1`
    Setting `process_zero=-1` or any other negative number replaces `0` with `-1`
    """
    task = 'Main'
    classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'dining table', 'dog', 'horse', 'motorbike', 'person', 'potted plant', 'sheep', 'sofa', 'train',
               'tvmonitor']
    def __init__(self, *args, replace_zero=-1, unpack=False, preloaded=None, train_test=True, **kwargs):
        self.replace_zero = replace_zero
        super().__init__(*args, unpack=unpack, preloaded=preloaded, train_test=train_test, **kwargs)

    def _process_targets(self, targets):
        np.place(targets, targets == 0, np.sign(self.replace_zero))
        labels = (targets > 0).astype(int)
        return labels

    def download(self, path):
        """ Download a dataset from the source web-site """
        self.download_archive(path)
        with tarfile.open(self.localname, "r") as archive:
            d = defaultdict(list)
            class_files = [os.path.join(self.SETS_PATH, self.task, name.replace(' ', '')) + '_trainval.txt'
                           for name in self.classes]
            for class_file in class_files:
                data = archive.extractfile(class_file).read()
                for row in data.decode().split('\n')[:-1]:
                    key = row.split()[0]
                    value = int(row.split()[1])
                    d[key].append(value)

            train_ids = self._extract_ids(archive, 'train')
            test_ids = self._extract_ids(archive, 'val')

            images = np.array([self._extract_image(archive, self._image_path(name))
                               for name in [*train_ids, *test_ids]], dtype=object)

            targets = np.array([d[self._name(name)] for name in [*train_ids, *test_ids]])
            labels = self._process_targets(targets)
            preloaded = images, labels

            train_len, test_len = len(train_ids), len(test_ids)
            index, train_index, test_index = self._infer_train_test_index(train_len, test_len)

            return preloaded, index, train_index, test_index
