import sys
import importlib

sys.modules['dataset'] = importlib.import_module('dataset.dataset')
