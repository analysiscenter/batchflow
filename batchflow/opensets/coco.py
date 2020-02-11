""" Contains COCO-Stuff dataset for Semantic Segmentation """
import os
from os.path import dirname, basename
from zipfile import ZipFile
import tempfile

import PIL
import tqdm
import logging
import requests
import numpy as np

from . import ImagesOpenset
from .. import FilesIndex, any_action_failed, parallel

logger = logging.getLogger('COCO')

class BaseCOCO(ImagesOpenset):

    TRAIN_IMAGES_URL = 'http://images.cocodataset.org/zips/train2017.zip' 
    TEST_IMAGES_URL = 'http://images.cocodataset.org/zips/val2017.zip'
    ALL_URLS = [TRAIN_IMAGES_URL, TEST_IMAGES_URL]

    def __init__(self, *args, unpack=True, preloaded=None, train_test=False, **kwargs):
        super().__init__(*args, preloaded=preloaded, train_test=train_test, **kwargs)

    def _post_fn(self, all_res, *args, **kwargs):
        if not self.load_to_ram:
            return self._gather_fi(all_res, *args, **kwargs)
        return self._gather_di(all_res, *args, **kwargs)

    @parallel(init='_get_from_urls', post='_post_fn', target='t')
    def download(self, url, content, path=None):
        """ Download archive"""
        if path is None:
            path = tempfile.gettempdir()
        filename = os.path.basename(url)
        localname = os.path.join(path, filename)
        if not os.path.isfile(localname):
            r = requests.get(self.SOURCE_URL, stream=True)
            file_size = int(r.headers['Content-Length'])
            chunk = 1
            chunk_size = 1024 * 1e3 #MBs
            num_bars = int(file_size / chunk_size)
            with open(localname, 'wb') as f:
                for chunk in tqdm.tqdm(r.iter_content(chunk_size=chunk_size), total=num_bars, unit='MB',
                                       desc=filename, leave=True):
                    f.write(chunk)

        if not self.load_to_ram: # working with FileIndex
            with ZipFile(localname, 'r') as archive:
                directory = '/COCOImages' if content else '/COCOMasks'
                extract_to = dirname(localname) + directory
                logger.info('Extracting %s', localname)
                archive.extractall(extract_to)
            return extract_to

        else: # working with DatasetIndex
            # load data to preloaded
            raise NotImplementedError
    

class COCOStuff(BaseCOCO):

    MASKS_URL = 'http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuffthingmaps_trainval2017.zip'
    ALL_URLS = [*BaseCOCO.ALL_URLS, MASKS_URL]

    def __init__(self, *args, load_to_ram=False, **kwargs):
        self.load_to_ram = load_to_ram
        super().__init__(*args,  **kwargs)

    @property
    def _get_from_urls(self):
        """ List of URLs and type of content (True - images, False - masks) """
        return [[self.ALL_URLS[i], i in [0, 1]] for i in range(len(self.ALL_URLS))]
    
    def _gather_fi(self, all_res, *args, **kwargs):
        _ = args, kwargs
        if any_action_failed(all_res):
            raise IOError('Could not download files:', all_res)

        self.train_images_dir = all_res[0] + '/train2017/'
        self.test_images_dir = all_res[1] + '/val2017/'
        self.train_masks_dir = all_res[2] + '/train2017/'
        self.test_masks_dir = all_res[2] + '/val2017/'
        return None, FilesIndex(path=self.train_images_dir + '*')

    @property
    def masks_directory(self):
        return self.train_masks_dir

