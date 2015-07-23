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
        self.shape = None
        self.orig_shape = None
        self.labels = None  # array of labeled results
        self.labels_filt = None  # array of labeled data that are filtered, e.g. filtered by area, compactness etc.
        self.objects = None  # array where each object has unique label
        self.labels_v = None
        self.n_rows = None
        self.n_cols = None
        self.n_slices = None

        self.lesions = None  # list of lesions, set it with Lesions.extract_lesions(self.labels)

        # self.data_vis = self.data  # visualized data, can be image data (data) or labels
        # self.data_vis_L = self.data  # visualized data, can be image data (data) or labels
        # self.data_vis_R = self.data

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

        # self.data = tools.windowing(data_dict['data3d'], level=self.win_level, width=self.win_width)
        self.data = data_dict['data3d']
        self.mask = data_dict['segmentation']

        self.voxel_size = data_dict['voxelsize_mm']
        # data = data_zoom(data, voxel_size, params['working_voxelsize_mm'])
        # mask = data_zoom(data_dict['segmentation'], voxel_size, params['working_voxelsize_mm'])

        if slice_idx != -1:
            self.data = self.data[slice_idx, :, :]
            self.mask = self.mask[slice_idx, :, :]

        self.data_vis = self.data

        self.orig_shape = self.data.shape
        self.shape = self.data.shape
        self.n_slices, self.n_rows, self.n_cols = self.orig_shape
        # self.labels = np.zeros(self.orig_shape)

        self.loaded = True

    # def display_im(self):
    #     self.data_vis = self.data
    #
    # def display_labels(self):
    #     # self.data_vis = self.labels
    #     self.data_vis = self.labels_filt
    #
    # def display_contours(self):
    #     self.data_vis = self.data
    #
    # def display_im_L(self):
    #     self.data_vis_L = self.data
    #
    # def display_labels_L(self):
    #     # self.data_vis = self.labels
    #     self.data_vis_L = self.labels_filt
    #
    # def display_contours_L(self):
    #     self.data_vis_L = self.data
    #
    # def display_im_R(self):
    #     self.data_vis_R = self.data
    #
    # def display_labels_R(self):
    #     # self.data_vis = self.labels
    #     self.data_vis_R = self.labels_filt
    #
    # def display_contours_R(self):
    #     self.data_vis_R = self.data