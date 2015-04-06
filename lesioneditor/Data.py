__author__ = 'root'

import numpy as np

import logging
logger = logging.getLogger(__name__)
logging.basicConfig()

import pickle

import tools

class Data:

    def __init__(self, data=None, mask=None, filename=None):
        self.data = data
        self.mask = mask
        self.voxel_size = None
        self.orig_shape = None
        self.labels = None
        self.n_rows = None
        self.n_cols = None
        self.n_slices = None

        self.filename = filename
        self.loaded = False

        self.win_level = 50
        self.win_width = 300


    def load_data(self, filename, slice_idx=-1):
        self.filename = filename
        fcontent = None
        try:
            import gzip
            f = gzip.open(self.filename, 'rb')
            fcontent = f.read()
            f.close()
        except Exception as e:
            logger.warning("Input gzip exception: " + str(e))
            f = open(self.filename, 'rb')
            fcontent = f.read()
            f.close()
        data_dict = pickle.loads(fcontent)

        self.data = tools.windowing(data_dict['data3d'], level=self.win_level, width=self.win_width)
        self.mask = data_dict['segmentation']

        self.voxel_size = data_dict['voxelsize_mm']
        # data = data_zoom(data, voxel_size, params['working_voxelsize_mm'])
        # mask = data_zoom(data_dict['segmentation'], voxel_size, params['working_voxelsize_mm'])

        if slice_idx != -1:
            self.data = self.data[slice_idx, :, :]
            self.mask = self.mask[slice_idx, :, :]

        self.orig_shape = self.data.shape
        self.n_slices, self.n_rows, self.n_cols = self.orig_shape
        self.labels = np.zeros(self.orig_shape)

        self.loaded = True
