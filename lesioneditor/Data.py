__author__ = 'root'

import numpy as np

import logging
logger = logging.getLogger(__name__)
logging.basicConfig()

import pickle

import tools

# definition of different views
VIEW_TABLE = {'axial': (2,1,0),
              'sagittal': (1,0,2),
              'coronal': (2,0,1)}

class Data(object):

    def __init__(self, data=None, mask=None, filename=None):
        self.data = data
        self.mask = mask
        self.voxel_size = None
        self.shape = None
        self.orig_shape = None
        self.__labels = None  # array of labeled results
        self.__labels_aview = None
        self.__labels_filt = None  # array of labeled data that are filtered, e.g. filtered by area, compactness etc.
        self.objects = None  # array where each object has unique label
        self.labels_v = None
        self.n_rows = None
        self.n_cols = None
        self.n_slices = None

        self.lesions = None  # list of lesions, set it with Lesions.extract_lesions(self.labels)

        # self.data_vis = self.data  # visualized data, can be image data (data) or labels
        # self.data_vis_L = self.data  # visualized data, can be image data (data) or labels
        # self.data_vis_R = self.data

        # @labels.setter
        # def labels(self, data):
        #     self._labels_aview = self._labels.transpose(self.act_transposition)

        # @labels.deleter
        # def labels(self):
        #     del self._labels
        #     del self._labels_aview

        self.filename = filename
        self.loaded = False

        self.win_level = 50
        self.win_width = 300

        # seting up views
        self.actual_view = 'axial'
        self.act_transposition = VIEW_TABLE[self.actual_view]
        if self.data is not None:
            self.data_aview = self.data.transpose(self.act_transposition)
        else:
            self.data_aview = None

    @property
    def labels(self):
        return self.__labels

    @labels.setter
    def labels(self, data):
        self.__labels = data
        self.__labels_aview = self.__labels.transpose(self.act_transposition)

    @labels.deleter
    def labels(self):
        del self.__labels
        del self.__labels_aview

    @property
    def labels_aview(self):
        return self.__labels_aview

    @labels_aview.setter
    def labels_aview(self, data):
        self.__labels_aview = data

    def set_labels(self, labels):
        self.__labels = labels
        self.__labels_aview = self.__labels.transpose(self.act_transposition)

    @property
    def labels_filt(self):
        return self.__labels_filt

    @labels_filt.setter
    def labels_filt(self, data):
        self.__labels_filt = data

    @labels_filt.deleter
    def labels_filt(self):
        del self.__labels_filt

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
            self.data = self.data[:, :, slice_idx]
            self.mask = self.mask[:, :, slice_idx]

        # data_s = 200 * np.triu(np.ones((100,120), dtype=np.int))
        # data = np.dstack((data_s, data_s))
        # self.data = np.rollaxis(data, 2, 0)
        # self.mask = np.ones_like(self.data)

        self.data_vis = self.data
        self.data_aview = self.data.transpose(self.act_transposition)

        self.orig_shape = self.data.shape
        # self.shape = self.data.shape
        self.n_slices, self.n_rows, self.n_cols = self.orig_shape
        self.shape = self.data.shape
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

if __name__ == '__main__':
    d = Data()
    d.labels = np.array([[[1, 1, 1]]])
    print d.labels
    # print d.labels_aview

    d.labels = np.array([[[2, 2, 2]]])
    print d.labels