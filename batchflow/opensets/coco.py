""" Contains COCO-Stuff dataset for Semantic Segmentation """
import os
import tempfile
import logging
from os.path import dirname, basename
from zipfile import ZipFile
from glob import glob

import tqdm
import requests
from PIL import Image

from . import ImagesOpenset
from .. import FilesIndex, any_action_failed, parallel, ImagesBatch, inbatch_parallel

logger = logging.getLogger('COCO')


class BaseCOCO(ImagesOpenset):

    TRAIN_IMAGES_URL = 'http://images.cocodataset.org/zips/train2017.zip'
    TEST_IMAGES_URL = 'http://images.cocodataset.org/zips/val2017.zip'
    ALL_URLS = [TRAIN_IMAGES_URL, TEST_IMAGES_URL]

    def __init__(self, *args, preloaded=None, **kwargs):
        super().__init__(*args, preloaded=preloaded, **kwargs)

    @parallel(init='_get_from_urls', post='_post_fn', target='t')
    def download(self, url, folder, train_val, path=None):
        """ Download archive"""
        logger.info('Downloading %s', url)
        if path is None:
            path = tempfile.gettempdir()
        filename = basename(url)
        localname = os.path.join(path, filename)
        if not os.path.isfile(localname):
            r = requests.get(url, stream=True)
            file_size = int(r.headers['Content-Length'])
            chunk = 1
            chunk_size = 1024 * 1000 #MBs
            num_bars = int(file_size / chunk_size)
            with open(localname, 'wb') as f:
                for chunk in tqdm.tqdm(r.iter_content(chunk_size=chunk_size), total=num_bars,
                                       unit='MB', desc=filename, leave=True):
                    f.write(chunk)
        return self._extract_if_not_exist(localname, folder, train_val)


class COCOSegmentationBatch(ImagesBatch):

    @inbatch_parallel(init='indices', post='_assemble')
    def _load_mask(self, ix, src, dst):
        fullpath = self._make_path(ix)
        train_val = fullpath.split('/')[-2]     # 'train2017' or 'val2017
        name_no_ext = ix.split('.')[0]          # filename wo extension
        path_to_mask = os.path.join(self._dataset.masks_directory, # /tmp/COCOMasks
                                    train_val, name_no_ext + '.' + self.formats[1])
        return Image.open(path_to_mask)

class COCOSegmentation(BaseCOCO):

    MASKS_URL = 'http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuffthingmaps_trainval2017.zip'
    ALL_URLS = [*BaseCOCO.ALL_URLS, MASKS_URL]

    def __init__(self, *args, batch_class=COCOSegmentationBatch, drop_grayscale=True, **kwargs):
        self.drop_grayscale = drop_grayscale
        super().__init__(*args, batch_class=COCOSegmentationBatch, **kwargs)

    @property
    def _get_from_urls(self):
        """ List of URL to download, folder where to extract,
        and indicator whether its train or val part"""
        return [[url, folder, train_val] for url, folder, train_val in zip(self.ALL_URLS,
                                                                           ['COCOImages', 'COCOImages', 'COCOMasks'],
                                                                           ['train2017', 'val2017', 'train2017'])]

    def _extract_archive(self, localname, extract_to):
        with ZipFile(localname, 'r') as archive:
            archive.extractall(extract_to)

    def _extract_if_not_exist(self, localname, folder, train_val):
        """ Extracts the arcive to the specific folder. Returns the path to this filder"""
        extract_to = os.path.join(dirname(localname), folder)
        path = os.path.join(extract_to, train_val)
        if os.path.isdir(path):
            pass
        else:
            self._extract_archive(localname, extract_to)
        return path

    def _rgb_images_paths(self, path):
        return  [filename for filename in glob(path + '/*')
                 if Image.open(filename).mode == 'RGB']

    def _post_fn(self, all_res, *args, **kwargs):
        _ = args, kwargs
        if any_action_failed(all_res):
            raise IOError('Could not download files:', all_res)

        if self.drop_grayscale:
            self._train_index = FilesIndex(path=self._rgb_images_paths(all_res[0])) # 10s for _rgb_images_paths(),
            self._test_index = FilesIndex(path=self._rgb_images_paths(all_res[1]))  # 270s for constructor
        else:
            self._train_index = FilesIndex(path=all_res[0] + '/*')
            self._test_index = FilesIndex(path=all_res[1] + '/*')

        self.masks_directory = dirname(all_res[2])
        return None, FilesIndex.concat(self._train_index, self._test_index)


class COCOObjectDetection(BaseCOCO):
    pass


class COCOKeyPoint(BaseCOCO):
    pass
