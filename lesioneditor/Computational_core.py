__author__ = 'tomas'


import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage.morphology as scindimor
import scipy.ndimage.measurements as scindimea
import scipy.ndimage.interpolation as scindiint
import scipy.ndimage as scindi
import scipy.stats as scista

import skimage.segmentation as skiseg
import skimage.morphology as skimor
import skimage.filter as skifil
import skimage.exposure as skiexp
import skimage.measure as skimea
import skimage.transform as skitra

import cv2
import pygco

import tools
import py3DSeedEditor
# from mayavi import mlab

import TumorVisualiser

from sklearn import metrics
from sklearn.cluster import KMeans

import pickle

import Viewer_3D

import cv2

import Data
import Lesion

import logging
logger = logging.getLogger(__name__)
logging.basicConfig()

class Computational_core():

    def __init__(self, fname, params, status_bar):
        # self.params = self.init_params()
        # self.params = params
        # self.models = None
        self.status_bar = status_bar

        # self.labels = None  # list of unique labels
        # self.filtered_idxs = None

        # ext_list = ('pklz', 'pickle')
        # self.fname = fname
        # self.data_1 = Data.Data()
        # self.data_2 = Data.Data()
        # self.actual_data = self.data_1
        # self.active_serie = 1

        # self.objects = list()  # list of segmented lesions

        # loading data - both series if provided
        # if len(self.fname) > 0:
        #     name = self.fname[0]
        #     if name.split('.')[-1] in ext_list:
        #         self.data_1.load_data(name)
        #
        #     else:
        #         msg = 'Wrong data type, supported extensions: ', ', '.join(ext_list)
        #         raise IOError(msg)
        # if len(self.fname) > 1:
        #     name = self.fname[1]
        #     if name.split('.')[-1] in ext_list:
        #         self.data_2.load_data(name)
        #     else:
        #         msg = 'Wrong data type, supported extensions: ', ', '.join(ext_list)
        #         raise IOError(msg)

        # smooth data if allowed
        # self.data_1.data = self.smooth_data(self.data_1.data)
        # self.data_2.data = self.smooth_data(self.data_2.data)

    def data_zoom(self, data, voxelsize_mm, working_voxelsize_mm):
        zoom = voxelsize_mm / (1.0 * working_voxelsize_mm)
        data_res = scindi.zoom(data, zoom, mode='nearest', order=1).astype(np.int16)
        return data_res

    def zoom_to_shape(self, data, shape):
        zoom = np.array(shape, dtype=np.float) / np.array(data.shape, dtype=np.float)
        data_res = scindi.zoom(data, zoom, mode='nearest', order=1).astype(np.int16)
        return data_res

    # def load_pickle_data(self, fname, slice_idx=-1):
    #     fcontent = None
    #     try:
    #         import gzip
    #         f = gzip.open(fname, 'rb')
    #         fcontent = f.read()
    #         f.close()
    #     except Exception as e:
    #         logger.warning("Input gzip exception: " + str(e))
    #         f = open(fname, 'rb')
    #         fcontent = f.read()
    #         f.close()
    #     data_dict = pickle.loads(fcontent)
    #
    #     data = tools.windowing(data_dict['data3d'], level=self.params['win_level'], width=self.params['win_width'])
    #
    #     mask = data_dict['segmentation']
    #
    #     voxel_size = data_dict['voxelsize_mm']
    #     # data = data_zoom(data, voxel_size, params['working_voxelsize_mm'])
    #     # mask = data_zoom(data_dict['segmentation'], voxel_size, params['working_voxelsize_mm'])
    #
    #
    #     if slice_idx != -1:
    #         data = data[slice_idx, :, :]
    #         mask = mask[slice_idx, :, :]
    #
    #     return data, mask, voxel_size

    def estimate_healthy_pdf(self, data, mask, params):
        perc = params['perc']
        k_std_l = params['k_std_h']
        simple_estim = params['healthy_simple_estim']
        show_me = params['show_healthy_pdf_estim']

        # data = tools.smoothing_tv(data.astype(np.uint8), tv_weight)
        ints = data[np.nonzero(mask)]
        hist, bins = skiexp.histogram(ints, nbins=256)
        if simple_estim:
            mu, sigma = scista.norm.fit(ints)
        else:
            ints = data[np.nonzero(mask)]

            n_pts = mask.sum()
            perc_in = n_pts * perc / 100

            peak_idx = np.argmax(hist)
            n_in = hist[peak_idx]
            win_width = 0

            while n_in < perc_in:
                win_width += 1
                n_in = hist[peak_idx - win_width:peak_idx + win_width].sum()

            idx_start = bins[peak_idx - win_width]
            idx_end = bins[peak_idx + win_width]
            inners_m = np.logical_and(ints > idx_start, ints < idx_end)
            inners = ints[np.nonzero(inners_m)]

            # liver pdf -------------
            mu = bins[peak_idx] + params['hack_healthy_mu']
            sigma = k_std_l * np.std(inners) + params['hack_healthy_sigma']

        mu = int(mu)
        sigma = int(sigma)
        rv = scista.norm(mu, sigma)

        if show_me:
            plt.figure()
            plt.subplot(211)
            plt.plot(bins, hist)
            plt.title('histogram with max peak')
            plt.hold(True)
            plt.plot([mu, mu], [0, hist.max()], 'g')
            ax = plt.axis()
            plt.axis([0, 256, ax[2], ax[3]])
            # plt.subplot(212), plt.plot(bins, rv_l.pdf(bins), 'g')
            x = np.arange(0, 256, 0.1)
            plt.subplot(212), plt.plot(x, rv.pdf(x), 'g')
            plt.hold(True)
            plt.plot(mu, rv.pdf(mu), 'go')
            ax = plt.axis()
            plt.axis([0, 256, ax[2], ax[3]])
            plt.title('estimated normal pdf of healthy parenchym')
            # plt.show()

        return rv

    def estimate_outlier_pdf(self, data, mask, rv_healthy, outlier_type, params):
        prob_w = params['prob_w']
        show_me = params['show_outlier_pdf_estim']

        if outlier_type == 'hypo':
            hack_mu = params['hack_hypo_mu']
            hack_sigma = params['hack_hypo_sigma']
        elif outlier_type == 'hyper':
            hack_mu = params['hack_hyper_mu']
            hack_sigma = params['hack_hyper_sigma']

        probs = rv_healthy.pdf(data) * mask
        # hist, bins = skiexp.histogram(probs, nbins=100)
        # plt.figure()
        # plt.plot(bins, hist)
        # plt.show()

        max_prob = rv_healthy.pdf(rv_healthy.mean())
        # mean_prob = np.mean(probs[np.nonzero(mask)])
        print 'max_prob = %.3f' % max_prob
        # print 'mean prob = %.3f' % mean_prob
        prob_t = prob_w * max_prob

        ints_out_m = probs < prob_t * mask

        # TumorVisualiser.run(data, ints_out_m, params['healthy_label'], params['hypo_label'], params['hyper_label'], slice_axis=0, disp_smoothed=True)

        ints_out = data[np.nonzero(ints_out_m)]
        hist, bins = skiexp.histogram(ints_out, nbins=256)

        if outlier_type == 'hypo':
            ints = ints_out[np.nonzero(ints_out < rv_healthy.mean())]
            # m = ints_out_m * (data < rv_healthy.mean())
            # TumorVisualiser.run(data, m, params['healthy_label'], params['hypo_label'], params['hyper_label'], slice_axis=0, disp_smoothed=True)
        elif outlier_type == 'hyper':
            ints = ints_out[np.nonzero(ints_out > rv_healthy.mean())]
            # m = ints_out_m * (data > rv_healthy.mean())
            # TumorVisualiser.run(data, m, params['healthy_label'], params['hypo_label'], params['hyper_label'], slice_axis=0, disp_smoothed=True)
        else:
            print 'Wrong outlier specification.'
            return

        mu, sigma = scista.norm.fit(ints)

        # hack for moving pdfs of hypo and especially of hyper furhter from the pdf of healthy parenchyma
        mu += hack_mu
        sigma += hack_sigma
        #-----------

        mu = int(mu)
        sigma = int(sigma)
        rv = scista.norm(mu, sigma)

        if show_me:
            plt.figure()
            plt.subplot(211)
            plt.plot(bins, hist)
            plt.title('histogram with max peak')
            plt.hold(True)
            plt.plot([mu, mu], [0, hist.max()], 'g')
            ax = plt.axis()
            plt.axis([0, 256, ax[2], ax[3]])
            # plt.subplot(212), plt.plot(bins, rv_l.pdf(bins), 'g')
            x = np.arange(0, 256, 0.1)
            plt.subplot(212), plt.plot(x, rv.pdf(x), 'g')
            plt.hold(True)
            plt.plot(mu, rv.pdf(mu), 'go')
            ax = plt.axis()
            plt.axis([0, 256, ax[2], ax[3]])
            plt.title('estimated normal pdf of %sdense obejcts' % outlier_type)
            plt.show()

        return rv

    def get_unaries(self, data, mask, models, params):
        rv_heal = models['rv_heal']
        rv_hyper = models['rv_hyper']
        rv_hypo = models['rv_hypo']
        # mu_heal = models['mu_heal']
        mu_heal = rv_heal.mean()

        if params['erode_mask']:
            if data.ndim == 3:
                mask = tools.eroding3D(mask, skimor.disk(5), slicewise=True)
            else:
                mask = skimor.binary_erosion(mask, np.ones((5, 5)))

        unaries_healthy = - rv_heal.logpdf(data) * mask
        if params['unaries_as_cdf']:
            unaries_hyper = - np.log(rv_hyper.cdf(data) * rv_heal.pdf(mu_heal)) * mask
            # removing zeros with second lowest value so the log(0) wouldn't throw a warning -
            tmp = 1 - rv_hypo.cdf(data)
            values = np.unique(tmp)
            tmp = np.where(tmp == 0, values[1], tmp)
            #-
            unaries_hypo = - np.log(tmp * rv_heal.pdf(mu_heal)) * mask
            unaries_hypo = np.where(np.isnan(unaries_hypo), 0, unaries_hypo)
        else:
            unaries_hyper = - rv_hyper.logpdf(data) * mask
            unaries_hypo = - rv_hypo.logpdf(data) * mask

        unaries = np.dstack((unaries_hypo.reshape(-1, 1), unaries_healthy.reshape(-1, 1), unaries_hyper.reshape(-1, 1)))
        unaries = unaries.astype(np.int32)

        # slice = 17
        # # slice = 6
        # plt.figure()
        # plt.imshow(unaries_hypo.reshape(data.shape)[slice,:,:], 'gray')
        # while True:
        #     pt = plt.ginput(1)
        #     print pt[0][1], pt[0][0], ' - ',
        #     print 'hypo = ', unaries_hypo.reshape(data.shape)[slice, int(pt[0][1]), int(pt[0][0])],
        #     print 'heal = ', unaries_healthy.reshape(data.shape)[slice, int(pt[0][1]), int(pt[0][0])],
        #     print 'hyper = ', unaries_hyper.reshape(data.shape)[slice, int(pt[0][1]), int(pt[0][0])],
        #     print ', int = ', data[slice, int(pt[0][1]), int(pt[0][0])]
        #
        # plt.figure()
        # plt.subplot(221), plt.imshow(unaries_hypo.reshape(data.shape)[6,:,:], 'gray'), plt.colorbar()
        # plt.subplot(222), plt.imshow(unaries_hyper.reshape(data.shape)[6,:,:], 'gray'), plt.colorbar()
        # plt.subplot(223), plt.imshow(unaries_healthy.reshape(data.shape)[6,:,:], 'gray'), plt.colorbar()
        # plt.show()

        # self.params['show_unaries'] = True
        if self.params['show_unaries']:
            ints = data[np.nonzero(mask)]
            hist, bins = skiexp.histogram(ints, nbins=256)
            x = np.arange(0, 255, 0.01)
            healthy = rv_heal.pdf(x)
            if params['unaries_as_cdf']:
                hypo = (1 - rv_hypo.cdf(x)) * rv_heal.pdf(mu_heal)
                hyper = rv_hyper.cdf(x) * rv_heal.pdf(mu_heal)
            else:
                hypo = rv_hypo.pdf(x)
                hyper = rv_hyper.pdf(x)

            plt.figure()
            plt.subplot(211)
            plt.plot(bins, hist)
            ax = plt.axis()
            plt.axis([0, 256, ax[2], ax[3]])
            plt.title('histogram of input data')
            plt.subplot(212)
            plt.plot(x, hypo, 'm')
            plt.hold(True)
            plt.plot(x, healthy, 'g')
            plt.plot(x, hyper, 'r')
            ax = plt.axis()
            plt.axis([0, 256, ax[2], ax[3]])
            plt.title('histogram of input data')
            plt.legend(['hypodense pdf', 'healthy pdf', 'hyperdense pdf'])
            plt.show()

        return unaries

    # def mayavi_visualization(self, res):
    #     ### Read the data in a numpy 3D array ##########################################
    #     #parenchym = np.logical_or(liver, vessels)
    #     parenchym = res > 0
    #     hypodense = res == 0
    #     hyperdense = res == 2
    #     data = np.where(parenchym, 1, 0)
    #     data = data.T
    #
    #     src = mlab.pipeline.scalar_field(data)
    #     src.spacing = [1, 1, 1]
    #
    #     data2 = np.where(hypodense, 1, 0)
    #     # data2 = hypodense
    #     data2 = data2.T
    #     src2 = mlab.pipeline.scalar_field(data2)
    #     src2.spacing = [1, 1, 1]
    #
    #     data3 = np.where(hyperdense, 1, 0)
    #     # data3 = hyperdense
    #     data3 = data3.T
    #     src3 = mlab.pipeline.scalar_field(data3)
    #     src3.spacing = [1, 1, 1]
    #
    #     #contours 6 ... cevy
    #     #contours 3 ... jatra
    #     #contours 10 ... jatra a cevy
    #     #mlab.pipeline.iso_surface(src, contours=3, opacity=0.1)
    #     #mlab.pipeline.iso_surface(src, contours=2, opacity=0.2)
    #     mlab.pipeline.iso_surface(src, contours=2, opacity=0.2, color=(0, 1, 0))
    #     mlab.pipeline.iso_surface(src2, contours=2, opacity=0.2, color=(1, 0, 0))
    #     mlab.pipeline.iso_surface(src3, contours=2, opacity=0.2, color=(0, 0, 1))
    #
    #     mlab.show()

    def get_compactness(self, labels):
        nlabels = labels.max() + 1
        # nlabels = len(np.unique(labels)) - 1
        eccs = np.zeros(nlabels)

        for lab in range(nlabels):
            obj = labels == lab
            if labels.ndim == 2:
                strel = np.ones((3, 3), dtype=np.bool)
                obj_c = skimor.binary_closing(obj, strel)
                if obj_c.sum() >= obj.sum():
                    obj = obj_c
            else:
                strel = np.ones((3, 3, 3), dtype=np.bool)
                obj_c = tools.closing3D(obj, strel)
                if obj_c.sum() >= obj.sum():
                    obj = obj_c

            if labels.ndim == 2:
                ecc = skimea.regionprops(obj)[0]['eccentricity']
            else:
                ecc = tools.get_zunics_compatness(obj)
                # if np.isnan(ecc):
                #     ecc = 0
            eccs[lab] = ecc

        return eccs

    def get_areas(self, labels):
        nlabels = labels.max() + 1
        areas = np.zeros(nlabels)
        for i in range(nlabels):
            obj = labels == i
            areas[i] = obj.sum()

        return areas

    # def filter_objects(self, feature_v, features, params):
    #     min_area = params['min_area']
    #     min_comp = params['min_compactness']
    #
    #     obj_ok = np.zeros(feature_v.shape, dtype=np.bool)
    #     area_idx = -1
    #     comp_idx = -1
    #     try:
    #         area_idx = features.index('area')
    #     except ValueError:
    #         pass
    #     try:
    #         comp_idx = features.index('compactness')
    #     except ValueError:
    #         pass
    #
    #     if area_idx != -1:
    #         obj_ok[:, area_idx] = feature_v[:, area_idx] > min_area
    #
    #     if comp_idx != -1:
    #         obj_ok[:, comp_idx] = feature_v[:, comp_idx] > min_comp
    #
    #     return obj_ok

    def smooth_data(self, data):
        if data is not None:
            # smoothing data
            print 'smoothing data...'
            if self.params['smoothing'] == 1:
                data = skifil.gaussian_filter(self.data, sigma=self.params['sigma'])
            elif self.params['smoothing'] == 2:
                data = tools.smoothing_bilateral(data, sigma_space=self.params['sigma_spatial'], sigma_color=self.params['sigma_range'], sliceId=0)
            elif self.params['smoothing'] == 3:
                data = tools.smoothing_tv(data, weight=self.params['weight'], sliceId=0)
            else:
                print '\tcurrently switched off'

        return data

    def calculate_intensity_models(self, data, mask):
        print 'calculating intensity models...'
        # liver pdf ------------
        rv_heal = self.estimate_healthy_pdf(data, mask, self.params)
        print '\tliver pdf: mu = ', rv_heal.mean(), ', sigma = ', rv_heal.std()
        # hypodense pdf ------------
        rv_hypo = self.estimate_outlier_pdf(data, mask, rv_heal, 'hypo', self.params)
        print '\thypodense pdf: mu = ', rv_hypo.mean(), ', sigma = ', rv_hypo.std()
        # hyperdense pdf ------------
        rv_hyper = self.estimate_outlier_pdf(data, mask, rv_heal, 'hyper', self.params)
        print '\thyperdense pdf: mu = ', rv_hyper.mean(), ', sigma = ', rv_hyper.std()

        models = dict()
        models['rv_heal'] = rv_heal
        models['rv_hypo'] = rv_hypo
        models['rv_hyper'] = rv_hyper

        return models

    def objects_filtration(self, selected_labels=None,
                           area=None,#min_area=0, max_area=np.Infinity,
                           density=None,#min_density=-np.Infinity, max_density=np.Infinity):
                           compactness=None):

        if self.actual_data.lesions is not None:
            lesions = self.actual_data.lesions[:]  # copy of the list
        else:
            return

        if area is not None:
            lesions = [x for x in lesions if area[0] <= x.area <= area[1] or x.priority == Lesion.PRIORITY_HIGH]

        if density is not None:
            lesions = [x for x in lesions if density[0] <= x.mean_density <= density[1] or x.priority == Lesion.PRIORITY_HIGH]

        if compactness is not None:
            if compactness > 1:
                compactness = float(compactness) / self.params['compactness_step']
            lesions = [x for x in lesions if compactness <= x.compactness or x.priority == Lesion.PRIORITY_HIGH]

        # geting labels of filtered objects
        self.filtered_idxs = [x.label for x in lesions]

        if selected_labels is not None:
            self.filtered_idxs = np.intersect1d(self.filtered_idxs, selected_labels)

        self.actual_data.labels_filt = np.where(self.actual_data.labels > self.params['bgd_label'], self.params['healthy_label'], self.params['bgd_label'])
        is_in = np.in1d(self.actual_data.labels, self.filtered_idxs).reshape(self.actual_data.labels.shape)
        self.actual_data.labels_filt = np.where(is_in, self.actual_data.labels, self.actual_data.labels_filt)
        pass

    def run(self):
        # slice_idx = self.params['slice_idx']
        alpha = self.params['alpha']
        beta = self.params['beta']
        hypo_lab = self.params['hypo_label']
        hyper_lab = self.params['hyper_label']

        # if not fname:
        #     _, self.data, self.mask = self.load_data(slice_idx)
        # else:
        #     self.data, self.mask, self.voxel_size = self.load_pickle_data(self.fname, self.params, slice_idx)
        # self.data = self.data.astype(np.uint8)

        # TumorVisualiser.run(data, mask, params['healthy_label'], params['hypo_label'], params['hyper_label'], slice_axis=0, disp_smoothed=True)

        # print 'estimating number of clusters ...',
        # print '\tcurrently switched off'
        # d = scindiint.zoom(data_o, 0.5)
        # m = skitra.resize(mask, np.array(mask.shape) * 0.5).astype(np.bool)
        # ints = d[np.nonzero(m)]
        # ints = ints.reshape((ints.shape[0], 1))
        # for i in range(2, 5):
        #     kmeans_model = KMeans(n_clusters=i, n_init=1).fit(ints)
        #     labels = kmeans_model.labels_
        #
        #     sc = metrics.silhouette_score(ints, labels, metric='euclidean')
        #     print '\tn_clusters = %i, score = %1.3f (best score=1, worse score=-1)' % (i, sc)

        # data_o = cv2.imread('/home/tomas/Dropbox/images/medicine/hypodense_bad2.png', 0).astype(np.float)

        # data_s = skifil.gaussian_filter(data_o, sigma)
        #
        # #imd = np.absolute(im - ims)
        # data_d = data_o - data_s
        #
        # data_d = np.where(data_d < 0, 0, data_d)

        # zooming the data
        print 'rescaling data ...',
        if self.params['zoom']:
            self.actual_data.data = self.data_zoom(self.actual_data.data, self.actual_data.voxel_size, self.params['working_voxel_size_mm'])
            self.actual_data.mask = self.data_zoom(self.actual_data.mask, self.actual_data.voxel_size, self.params['working_voxel_size_mm'])
        else:
            data = tools.resize3D(self.actual_data.data, self.params['scale'], sliceId=0)
            mask = tools.resize3D(self.actual_data.mask, self.params['scale'], sliceId=0)
        print 'ok'
        # data = data.astype(np.uint8)

        # calculating intensity models if necesarry
        print 'estimating color models ...'
        if not self.models:
            self.models = self.calculate_intensity_models(data, mask)
            print 'ok'
        else:
            print 'already done'

        print 'calculating unary potentials ...',
        self.status_bar.showMessage('Calculating unary potentials...')
        # create unaries
        # unaries = data_d
        # # as we convert to int, we need to multipy to get sensible values
        # unaries = (1 * np.dstack([unaries, -unaries]).copy("C")).astype(np.int32)
        self.unaries = beta * self.get_unaries(data, mask, self.models, self.params)
        n_labels = self.unaries.shape[2]
        print 'ok'

        print 'calculating pairwise potentials ...',
        self.status_bar.showMessage('Calculating pairwise potentials...')
        # create potts pairwise
        self.pairwise = -alpha * np.eye(n_labels, dtype=np.int32)
        print 'ok'

        print 'deriving graph edges ...',
        self.status_bar.showMessage('Deriving graph edges...')
        # use the gerneral graph algorithm
        # first, we construct the grid graph
        inds = np.arange(data.size).reshape(data.shape)
        if data.ndim == 2:
            horz = np.c_[inds[:, :-1].ravel(), inds[:, 1:].ravel()]
            vert = np.c_[inds[:-1, :].ravel(), inds[1:, :].ravel()]
            self.edges = np.vstack([horz, vert]).astype(np.int32)
        elif data.ndim == 3:
            # horz = np.c_[inds[:, :-1].ravel(), inds[:, 1:].ravel()]
            # vert = np.c_[inds[:-1, :].ravel(), inds[1:, :].ravel()]
            horz = np.c_[inds[:, :, :-1].ravel(), inds[:, :, 1:].ravel()]
            vert = np.c_[inds[:, :-1, :].ravel(), inds[:, 1:, :].ravel()]
            dept = np.c_[inds[:-1, :, :].ravel(), inds[1:, :, :].ravel()]
            self.edges = np.vstack([horz, vert, dept]).astype(np.int32)
        # deleting edges with nodes outside the mask
        nodes_in = np.ravel_multi_index(np.nonzero(mask), data.shape)
        rows_inds = np.in1d(self.edges, nodes_in).reshape(self.edges.shape).sum(axis=1) == 2
        self.edges = self.edges[rows_inds, :]
        print 'ok'

        # plt.show()

        # un = unaries[:, :, 0].reshape(data_o.shape)
        # un = unaries[:, :, 1].reshape(data_o.shape)
        # un = unaries[:, :, 2].reshape(data_o.shape)
        # py3DSeedEditor.py3DSeedEditor(un).show()

        print 'calculating graph cut ...',
        self.status_bar.showMessage('Calculating graph cut...')
        # we flatten the unaries
        # result_graph = pygco.cut_from_graph(edges, unaries.reshape(-1, 2), pairwise)
        # print 'tu: ', unaries.reshape(-1, n_labels).shape
        result_graph = pygco.cut_from_graph(self.edges, self.unaries.reshape(-1, n_labels), self.pairwise)
        labels = result_graph.reshape(data.shape) + 1  # +1 to shift the first class to label number 1

        labels = np.where(mask, labels, self.params['bgd_label'])

        # zooming to the original size
        if self.params['zoom']:
            self.actual_data.labels = self.zoom_to_shape(labels, self.actual_data.orig_shape)
        else:
            # self.actual_data.labels = tools.resize3D(labels, scale=1. / self.params['scale'], sliceId=0)
            # self.actual_data.labels2 = skitra.resize(labels, self.actual_data.orig_shape, mode='nearest', preserve_range=True).astype(np.int64)
            self.actual_data.labels = tools.resize3D(labels, shape=self.actual_data.orig_shape, sliceId=0)#.astype(np.int64)

        print 'ok'
        self.status_bar.showMessage('Done')

        # debug visualization
        # self.viewer = Viewer_3D.Viewer_3D(self.res, range=True)
        # self.viewer.show()

        print 'extracting objects ...',
        self.status_bar.showMessage('Extracting objects ...'),
        labels_tmp = np.where(self.actual_data.labels == self.params['healthy_label'], self.params['bgd_label'], self.actual_data.labels)  # because we can set only one label as bgd
        self.actual_data.objects = skimea.label(labels_tmp, background=self.params['bgd_label'])
        self.actual_data.lesions = Lesion.extract_lesions(self.actual_data.objects, self.actual_data.data)
        # areas = [x.area for x in self.actual_data.lesions]
        self.status_bar.showMessage('Done')
        print 'ok'

        print 'initial filtration ...',
        self.objects_filtration(area=(self.params['min_area'], self.params['max_area']),
                                density=(self.params['min_density'], self.params['max_density']),
                                compactness=self.params['min_compactness'])
        print 'ok'
        self.status_bar.showMessage('Done')


        # TumorVisualiser.run(self.data, self.res, self.params['healthy_label'], self.params['hypo_label'], self.params['hyper_label'], slice_axis=0)

        # mayavi_visualization(res)

#-------------------------------------------------------------
if __name__ == '__main__':

    fname = ''

    # 2 hypo, 1 on the border --------------------
    # arterial 0.6mm - bad
    # fname = '/home/tomas/Data/liver_segmentation_06mm/tryba/data_other/org-exp_183_46324212_arterial_0.6_B30f-.pklz'
    # venous 0.6mm - good
    # fname = '/home/tomas/Data/liver_segmentation_06mm/tryba/data_other/org-exp_183_46324212_venous_0.6_B20f-.pklz'
    # venous 5mm - ok, but wrong approach
    # fname = '/home/tomas/Data/liver_segmentation/tryba/data_other/org-exp_183_46324212_venous_5.0_B30f-.pklz'

    # hypo in venous -----------------------
    # arterial - bad
    # fname = '/home/tomas/Data/liver_segmentation_06mm/tryba/data_other/org-exp_186_49290986_venous_0.6_B20f-.pklz'
    # venous - good
    fname = '/home/tomas/Data/liver_segmentation_06mm/tryba/data_other/org-exp_186_49290986_arterial_0.6_B20f-.pklz'

    # hyper, 1 on the border -------------------
    # arterial 0.6mm - not that bad
    # fname = '/home/tomas/Data/liver_segmentation_06mm/hyperdenzni/org-exp_239_61293268_DE_Art_Abd_0.75_I26f_M_0.5-.pklz'
    # venous 5mm - bad
    # fname = '/home/tomas/Data/liver_segmentation_06mm/hyperdenzni/org-exp_239_61293268_DE_Ven_Abd_0.75_I26f_M_0.5-.pklz'

    # shluk -----------------
    # arterial 5mm
    # fname = '/home/tomas/Data/liver_segmentation/tryba/data_other/org-exp_180_49509315_arterial_5.0_B30f-.pklz'
    # fname = '/home/tomas/Data/liver_segmentation_06mm/tryba/data_other/org-exp_180_49509315_arterial_0.6_B20f-.pklz'

    # targeted
    # arterial 0.6mm - bad
    # fname = '/home/tomas/Data/liver_segmentation_06mm/hyperdenzni/org-exp_238_54280551_Abd_Arterial_0.75_I26f_3-.pklz'
    # venous 0.6mm - b  ad
    # fname = '/home/tomas/Data/liver_segmentation_06mm/hyperdenzni/org-exp_238_54280551_Abd_Venous_0.75_I26f_3-.pklz'

    cc = Computational_core(fname)
    cc.run()