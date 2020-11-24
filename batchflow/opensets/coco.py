""" Datasets for COCO challenge tasks, http://cocodataset.org/#home.
    Currently contains only the dataset for Semantic Segmentation. """

import os
import logging
import tempfile
from glob import glob
from zipfile import ZipFile
from os.path import dirname, basename

import tqdm
import requests
from PIL import Image

from . import ImagesOpenset
from .. import FilesIndex, any_action_failed, parallel

logger = logging.getLogger('COCO')


class BaseCOCO(ImagesOpenset):
    """ Base class for COCO datasets. """

    TRAIN_IMAGES_URL = 'http://images.cocodataset.org/zips/train2017.zip'
    TEST_IMAGES_URL = 'http://images.cocodataset.org/zips/val2017.zip'

    def _rgb_images_paths(self, path):
        """Find RGB images(avoid grayscale) in the folder and return their paths. """
        return  [filename for filename in glob(path + '/*') if Image.open(filename).mode == 'RGB']

    @parallel(init='_get_from_urls', post='_post_fn', target='t')
    def download(self, url, content, train_val, path=None):
        """ Download the archives and extract their contents in a parallel manner.
        Returns the path to the folder where the archive extracted.
        Set of URL's to download from is defined in the `_get_from_urls` method.
        The aggregation of the content from all archives is performed in `_post_fn` method.
        """
        logger.info('Downloading %s', url)
        if path is None:
            path = tempfile.gettempdir()
        filename = basename(url)
        localname = os.path.join(path, filename)
        if not os.path.isfile(localname):
            r = requests.get(url, stream=True)
            file_size = int(r.headers['Content-Length'])
            chunk_size = 1024 * 1000 #MBs
            num_bars = int(file_size / chunk_size)
            # downloading
            with open(localname, 'wb') as f:
                for chunk in tqdm.tqdm(r.iter_content(chunk_size=chunk_size), total=num_bars,
                                       unit='MB', desc=filename, leave=True, disable=self.disable_tqdm):
                    f.write(chunk)

        folder_to_extract = os.path.join(dirname(localname), content)
        #check that root folder from the archive already exists to avoid extracting, as its time consuming
        path_to_extracted = os.path.join(folder_to_extract, train_val)
        if os.path.isdir(path_to_extracted):
            pass
        else:
            # extracting
            with ZipFile(localname, 'r') as archive:
                archive.extractall(folder_to_extract)
        return path_to_extracted


class COCOSegmentation(BaseCOCO):
    """ The dataset for COCOStuff challenge for pixel-wise segmentation. Total size 18GB.
    Contains 118060 train and 4990 test images/masks.

    Notes
    -----
    - Datasets contain both grayscale and colored images
      Argument `drop_grayscale` controls whether grayscale images should be dropped.
    """

    MASKS_URL = 'http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuffthingmaps_trainval2017.zip'

    def __init__(self, *args, bar=False, drop_grayscale=True, **kwargs):
        self.disable_tqdm = not bar
        self.drop_grayscale = drop_grayscale
        super().__init__(*args, **kwargs)

    @property
    def _get_from_urls(self):
        """ List of URL to download from, folder where to extract, and indicator whether its train or val part. """
        iterator = zip([self.TRAIN_IMAGES_URL, self.TEST_IMAGES_URL, self.MASKS_URL, self.MASKS_URL],
                       ['COCOImages', 'COCOImages', 'COCOMasks', 'COCOMasks'],
                       ['train2017', 'val2017', 'train2017', 'val2017'])
        return [[url, content, train_val] for url, content, train_val in iterator]

    def _post_fn(self, all_res, *args, **kwargs):
        _ = args, kwargs
        if any_action_failed(all_res):
            raise IOError('Could not download files:', all_res)

        if self.drop_grayscale:
            train_index = FilesIndex(path=self._rgb_images_paths(all_res[0]), no_ext=True)
            test_index = FilesIndex(path=self._rgb_images_paths(all_res[1]), no_ext=True)
        else:
            train_index = FilesIndex(path=all_res[0] + '/*', no_ext=True)
            test_index = FilesIndex(path=all_res[1] + '/*', no_ext=True)
        index = FilesIndex.concat(train_index, test_index)

        # store the paths to the folders with masks as attributes
        setattr(self, 'path_train_masks', all_res[2])
        setattr(self, 'path_test_masks', all_res[3])
        return None, index, train_index, test_index
