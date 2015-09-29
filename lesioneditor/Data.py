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
        self.mask = mask
        self.voxel_size = None
        self.slab = None
        self.shape = None
        self.orig_shape = None
        self.objects = None  # array where each object has unique label
        # self.object_centroids = None
        self.labels_v = None
        self.n_rows = None
        self.n_cols = None
        self.n_slices = None
        self.models = None  # color models

        # properties
        self.__data = data
        self.__labels = None  # array of labeled results (only 3 classes: healthy, hypodense and hyperdense
        self.__labels_aview = None
        self.__labels_filt = None  # array of labeled data that are filtered, e.g. filtered by area, compactness etc.
        self.__labels_filt_aview = None
        self.__object_centers = None  # array of object centers
        self.__object_centers_filt = None
        self.__object_centers_list = list()  # list of object centers
        self.__lesions = list()  # list of lesions, set it with Lesions.extract_lesions(self.labels)
        self.__lesions_filt = list()

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
    def data(self):
        return self.__data

    @data.setter
    def data(self, x):
        self.__data = x
        self.labels = np.zeros(x.shape)
        self.objects = np.zeros(x.shape)
        self.object_centers = np.zeros(x.shape)
        self.orig_shape = x.shape
        self.shape = x.shape

    @property
    def labels(self):
        return self.__labels

    @labels.setter
    def labels(self, x):
        self.__labels = x
        self.__labels_aview = self.__labels.transpose(self.act_transposition)
        self.__labels_filt = x
        self.__labels_filt_aview = self.__labels_filt.transpose(self.act_transposition)

    @labels.deleter
    def labels(self):
        del self.__labels
        del self.__labels_aview

    @property
    def labels_aview(self):
        return self.__labels_aview

    @labels_aview.setter
    def labels_aview(self, x):
        self.__labels_aview = x

    @property
    def labels_filt(self):
        return self.__labels_filt

    @labels_filt.setter
    def labels_filt(self, x):
        self.__labels_filt = x
        self.__labels_filt_aview = x.transpose(self.act_transposition)

        # print 'setting lesion_filt ...',
        # self.__lesions_filt = [x for x in self.lesions if x.label in np.unique(self.__labels_filt)]
        # print 'done'

        # print 'setting object_centers_filt...',
        # self.__object_centers_filt = np.zeros(self.objects.shape, dtype=np.bool)
        # for i in self.__lesions_filt:
        #     try:
        #         idx = (np.round(i.center)).astype(np.int)
        #         self.__object_centers_filt[idx[0], idx[1], idx[2]] = 1
        #     except:
        #         pass
        # print 'done'

    @labels_filt.deleter
    def labels_filt(self):
        del self.__labels_filt

    @property
    def labels_filt_aview(self):
        return self.__labels_filt_aview

    @labels_filt_aview.setter
    def labels_filt_aview(self, x):
        self.__labels_filt_aview = x

    # @property
    # def user_seeds(self):
    #     return self.__user_seeds

    @property
    def lesions(self):
        return self.__lesions

    @lesions.setter
    def lesions(self, x):
        self.__lesions = x
        self.__lesions_filt = x[:]
        self.__object_centers_list = [l.center for l in self.lesions]
        self.__object_centers = np.zeros(self.objects.shape, dtype=np.bool)
        for i in x:
            try:
                idx = (np.round(i.center)).astype(np.int)
                self.__object_centers[idx[0], idx[1], idx[2]] = 1
            except:
                pass

    @property
    def lesions_filt(self):
        return self.__lesions_filt

    @lesions_filt.setter
    def lesions_filt(self, x):
        self.__lesions_filt = x

        centers = [l.center for l in self.lesions_filt]

        centers_filt = np.zeros(self.objects.shape, dtype=np.bool)
        for i in centers:
            try:
                idx = np.round(i).astype(np.int)
                centers_filt[idx[0], idx[1], idx[2]] = 1
            except:
                pass
        self.__object_centers_filt = centers_filt

    @property
    def object_centers(self):
        return self.__object_centers

    @object_centers.setter
    def object_centers(self, x):
        self.__object_centers = x

    @property
    def object_centers_filt(self):
        return self.__object_centers_filt

    @object_centers_filt.setter
    def object_centers_filt(self, x):
        self.__object_centers_filt = x

    @property
    def object_centers_list(self):
        return self.__object_centers_list

    @object_centers_list.setter
    def object_centers_list(self, x):
        self.__object_centers_list = x

    def append_lesion(self, les):
        self.__lesions.append(les)
        self.__object_centers[les.center[0], les.center[1], les.center[2]] = 1
        self.__object_centers_list.append(les.center)

    # def load_data(self, filename, slice_idx=-1):
    #     self.filename = filename
    #     fcontent = None
    #     try:
    #         import gzip
    #         f = gzip.open(self.filename, 'rb')
    #         fcontent = f.read()
    #         f.close()
    #     except Exception as e:
    #         logger.warning("Input gzip exception: " + str(e))
    #         f = open(self.filename, 'rb')
    #         fcontent = f.read()
    #         f.close()
    #     data_dict = pickle.loads(fcontent)
    #
    #     self.data = data_dict['data3d']
    #     self.mask = data_dict['segmentation']
    #
    #     self.voxel_size = data_dict['voxelsize_mm']
    #
    #     if slice_idx != -1:
    #         self.data = self.data[:, :, slice_idx]
    #         self.mask = self.mask[:, :, slice_idx]
    #
    #     # data_s = 200 * np.triu(np.ones((100,120), dtype=np.int))
    #     # data = np.dstack((data_s, data_s))
    #     # self.data = np.rollaxis(data, 2, 0)
    #     # self.mask = np.ones_like(self.data)
    #
    #     self.data_vis = self.data
    #     self.data_aview = self.data.transpose(self.act_transposition)
    #
    #     self.orig_shape = self.data.shape
    #     self.n_slices, self.n_rows, self.n_cols = self.orig_shape
    #     self.shape = self.data.shape
    #
    #     self.loaded = True

    def create_data(self, datap, filename):
        if datap is None and filename is None:
            raise AttributeError('No data nor filename given.')

        # if filename is not None:
        #     self.filename = filename
        #     fcontent = None
        #     try:
        #         import gzip
        #         f = gzip.open(self.filename, 'rb')
        #         fcontent = f.read()
        #         f.close()
        #     except Exception as e:
        #         logger.warning("Input gzip exception: " + str(e))
        #         f = open(self.filename, 'rb')
        #         fcontent = f.read()
        #         f.close()
        #     datap = pickle.loads(fcontent)
        # else:
        self.filename = filename

        self.data = datap['data3d']
        self.mask = datap['segmentation']
        self.voxel_size = datap['voxelsize_mm']
        self.slab = datap['slab']

        self.data_vis = self.data
        self.data_aview = self.data.transpose(self.act_transposition)

        self.orig_shape = self.data.shape
        self.n_slices, self.n_rows, self.n_cols = self.orig_shape
        self.shape = self.data.shape

        self.loaded = True

if __name__ == '__main__':
    d = Data()
    d.labels = np.array([[[1, 1, 1]]])
    print d.labels

    d.labels = np.array([[[2, 2, 2]]])
    print d.labels