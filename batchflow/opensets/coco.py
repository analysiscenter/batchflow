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
from .. import FilesIndex, any_action_failed, parallel, ImagesBatch, inbatch_parallel

logger = logging.getLogger('COCO')


class BaseCOCO(ImagesOpenset):

    TRAIN_IMAGES_URL = 'http://images.cocodataset.org/zips/train2017.zip' 
    TEST_IMAGES_URL = 'http://images.cocodataset.org/zips/val2017.zip'
    ALL_URLS = [TRAIN_IMAGES_URL, TEST_IMAGES_URL]

    def __init__(self, *args, preloaded=None, **kwargs):
        super().__init__(*args, preloaded=preloaded, **kwargs)

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
        
        return self.extract_all(localname, content)


class COCOSegmentationBatch(ImagesBatch):

    @inbatch_parallel(init='indices', post='_assemble')
    def _load_mask(self, ix, src, dst):
        fullpath = self._make_path(ix)
        name_no_ext = ix.split('.')[0]
        part = fullpath.split('/')[-2]
        path_to_mask = os.path.join(self._dataset.masks_directory, part, name_no_ext) + '.png'
        return PIL.Image.open(path_to_mask)


class COCOSegmentation(BaseCOCO):

    MASKS_URL = 'http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuffthingmaps_trainval2017.zip'
    ALL_URLS = [*BaseCOCO.ALL_URLS, MASKS_URL]

    def __init__(self, *args, batch_class=COCOSegmentationBatch, train_test=True, load_to_ram=False, **kwargs):
        self.load_to_ram = load_to_ram
        self.masks_directory = None
        super().__init__(*args, batch_class=COCOSegmentationBatch, train_test=train_test, **kwargs)

    @property
    def _get_from_urls(self):
        """ List of URLs and type of content (0 - train images, 1 - test images, 2 - train+test masks) """
        return [[self.ALL_URLS[i], i] for i in range(len(self.ALL_URLS))]
    
    def _extract_archive(self, localname, extract_to):
        with ZipFile(localname, 'r') as archive:
            archive.extract_all(extract_to)

    def extract_all(self, localname, content):
        directory = '/COCOImages'
        extract_to = dirname(localname) + directory
        if content == 0:
            path =  os.path.join(extract_to, 'train2017')
            if os.path.isdir(path):
                pass
            else:
                self._extract_archive(localname, extract_to)
        elif content == 1:
            path = os.path.join(extract_to, 'val2017')
            if os.path.isdir(path):
                pass
            else:
                self._extract_archive(localname, extract_to)
        else:
            directory = '/COCOMasks'
            extract_to = dirname(localname) + directory
            self.masks_directory = extract_to
            path = tuple([os.path.join(extract_to, folder_name) for folder_name in ['train2017', 'val2017']])
            if all([os.path.isdir(p) for p in path]):
                pass
            else:
                self._extract_archive(localname, extract_to)
        return path
                    
    
    def _gather_fi(self, all_res, *args, **kwargs):
        if any_action_failed(all_res):
            raise IOError('Could not download files:', all_res)

        self._train_index = FilesIndex(path=all_res[0] + '/*')
        self._test_index = FilesIndex(path=all_res[1] + '/*')
        return None, FilesIndex(path=[all_res[0] + '/*', all_res[1] + '/*'])


class COCOObjectDetection(BaseCOCO):
    pass


class COCOKeyPoint(BaseCOCO):
    pass


