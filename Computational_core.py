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

import logging
logger = logging.getLogger(__name__)
logging.basicConfig()

class Computational_core():

    def __init__(self, fname, params, status_bar):
        # self.params = self.init_params()
        self.params = params
        self.models = None
        self.status_bar = status_bar

        self.labels = None
        self.filtered_idxs = None

        ext_list = ('pklz', 'pickle')
        self.fname = fname
        if self.fname.split('.')[-1] in ext_list:
            self.data, self.mask, self.voxel_size = self.load_pickle_data(self.fname)
        else:
            msg = 'Wrong data type, supported extensions: ', ', '.join(ext_list)
            raise IOError(msg)
        self.orig_shape = self.data.shape

        # smooth data if allowed
        self.smooth_data()


    # def init_params(self):
    #     params = dict()
    #     params['slice_idx'] = -1
    #     # params['sigma'] = 10  # sigma for gaussian blurr
    #     params['alpha'] = 3  # weightening parameter for pairwise term
    #     params['beta'] = 1  # weightening parameter for unary term
    #     params['perc'] = 0.3  # what portion of liver parenchym around peak is used to calculate std of liver normal pdf
    #     params['k_std_h'] = 3  # weightening parameter for sigma of normal distribution of healthy parenchym
    #     params['k_std_t'] = 3  # weightening parameter for sigma of normal distribution of tumor
    #     # params['tv_weight'] = 0.05  # weighting parameter for total variation filter
    #     params['healthy_simple_estim'] = False  # simple healthy parenchym pdf estimation from all data
    #     params['prob_w'] = 0.0001
    #
    #     params['working_voxelsize_mm'] = 2  # size of voxels that will be used in computation
    #
    #     # data smoothing
    #     # 0 ... no smoothing
    #     # 1 ... gaussian blurr, param = sigma
    #     # 2 ... bilateral filter, param = sigma_range (0.05)
    #     # 3 ... total variation filter, param = weight (0.1)
    #     params['smoothing'] = -1
    #     params['sigma'] = 1
    #     params['sigma_range'] = 0.05
    #     params['sigma_spatial'] = 15
    #     params['weight'] = 0.05
    #
    #     params['win_width'] = 350  # width of window for visualising abdomen
    #     params['win_level'] = 50  # level of window for visualising abdomen
    #
    #     params['unaries_as_cdf'] = True  # whether to estimate the prob. model of outliers as cumulative density function
    #
    #     # These are not necessary now - user can edit the color model in the GUI.
    #     # However, using it in automated mode can be usefull.
    #     params['hack_hypo_mu'] = -0  # hard move of mean of hypodense pdf to the left
    #     params['hack_hypo_sigma'] = 0  # hard widening of sigma of hypodense pdf
    #     params['hack_hyper_mu'] = -0 #5  # hard move of mean of hyperdense pdf to the right
    #     params['hack_hyper_sigma'] = 0 #5  # hard widening of sigma of hyperdense pdf
    #     params['hack_healthy_mu'] = -0 #5  # hard move of mean of healthy pdf to the right
    #     params['hack_healthy_sigma'] = 0 #5  # hard widening of sigma of healthy pdf
    #
    #     params['show_healthy_pdf_estim'] = False
    #     params['show_outlier_pdf_estim'] = False
    #     params['show_estimated_pdfs'] = False
    #     params['show_unaries'] = False
    #
    #     params['hypo_label'] = 0  # label of hypodense objects
    #     params['healthy_label'] = 1
    #     params['hyper_label'] = 2  # label of hyperdense objects
    #
    #     params['filtration'] = False  # whether to filtrate or not
    #     params['min_area'] = 20
    #     params['min_compactness'] = 0.2
    #
    #     params['erode_mask'] = True
    #
    #     return params

#-------------------------------------------------------------
    def load_data(self, slice_idx=-1):
        print 'TODO: must be recoded to be consistent with outputs given by self.load_pickle_data(...)'
    # TODO: must be recoded to be consistent with outputs given by self.load_pickle_data(...)
    #     # 33 ... hypodense
    #     # 41 ... small tumor
    #     # 138 ... hyperdense
    #
    #     # labels = np.load('label_im.npy')
    #     data = np.load('input_data.npy')
    #     o_data = np.load('input_orig_data.npy')
    #     mask = np.load('mask.npy')
    #
    #     # to be sure that the mask is only binary
    #     mask = np.where(mask > 0, 1, 0)
    #
    #     if slice_idx != -1:
    #         data_s = data[slice_idx, :, :]
    #         o_data_s = o_data[slice_idx, :, :]
    #         mask_s = mask[slice_idx, :, :]
    #
    #         data, _ = tools.crop_to_bbox(data_s, mask_s)
    #         o_data, _ = tools.crop_to_bbox(o_data_s, mask_s)
    #         mask, _ = tools.crop_to_bbox(mask_s, mask_s)
    #
    #     # plt.figure()
    #     # plt.subplot(131), plt.imshow(data_bbox, 'gray')
    #     # plt.subplot(132), plt.imshow(o_data_bbox, 'gray')
    #     # plt.subplot(133), plt.imshow(mask_bbox, 'gray')
    #     # plt.show()
    #
    #     return data, o_data, mask

    def data_zoom(self, data, voxelsize_mm, working_voxelsize_mm):
        zoom = voxelsize_mm / (1.0 * working_voxelsize_mm)
        data_res = scindi.zoom(data, zoom, mode='nearest', order=1).astype(np.int16)
        return data_res


    def zoom_to_shape(self, data, shape):
        zoom = np.array(shape, dtype=np.float) / np.array(self.data.shape, dtype=np.float)
        data_res = scindi.zoom(data, zoom, mode='nearest', order=1).astype(np.int16)
        return data_res


    def load_pickle_data(self, fname, slice_idx=-1):
        fcontent = None
        try:
            import gzip
            f = gzip.open(fname, 'rb')
            fcontent = f.read()
            f.close()
        except Exception as e:
            logger.warning("Input gzip exception: " + str(e))
            f = open(fname, 'rb')
            fcontent = f.read()
            f.close()
        data_dict = pickle.loads(fcontent)

        data = tools.windowing(data_dict['data3d'], level=self.params['win_level'], width=self.params['win_width'])

        mask = data_dict['segmentation']

        voxel_size = data_dict['voxelsize_mm']
        # data = data_zoom(data, voxel_size, params['working_voxelsize_mm'])
        # mask = data_zoom(data_dict['segmentation'], voxel_size, params['working_voxelsize_mm'])


        if slice_idx != -1:
            data = data[slice_idx, :, :]
            mask = mask[slice_idx, :, :]

            # data, _ = tools.crop_to_bbox(data_s, mask_s)
            # mask, _ = tools.crop_to_bbox(mask_s, mask_s)

        # plt.figure()
        # plt.subplot(121), plt.imshow(data_bbox, 'gray')
        # plt.subplot(122), plt.imshow(mask_bbox, 'gray')
        # plt.show()

        return data, mask, voxel_size


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

        return mu, sigma, rv


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

        return mu, sigma, rv


    def get_unaries(self, data, mask, models, params):
        rv_heal = models['rv_heal']
        rv_hyper = models['rv_hyper']
        rv_hypo = models['rv_hypo']
        # mu_heal = models['mu_heal']
        mu_heal = self.mu_heal

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
            plt.plot(x, healthy, 'g')
            plt.plot(x, hyper, 'r')
            ax = plt.axis()
            plt.axis([0, 256, ax[2], ax[3]])
            plt.title('histogram of input data')
            plt.legend(['hypodense pdf', 'healthy pdf', 'hyperdense pdf'])
            # plt.show()

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


    def smooth_data(self):
        # smoothing data
        print 'smoothing data...'
        if self.params['smoothing'] == 1:
            self.data = skifil.gaussian_filter(self.data, )
        elif self.params['smoothing'] == 2:
            self.data = tools.smoothing_bilateral(self.data, sigma_space=self.params['sigma_spatial'], sigma_color=self.params['sigma_range'], sliceId=0)
        elif self.params['smoothing'] == 3:
            self.data = tools.smoothing_tv(self.data, weight=self.params['weight'], sliceId=0)
        else:
            print '\tcurrently switched off'


    def calculate_intensity_models(self):
        print 'calculating intensity models...'
        # liver pdf ------------
        self.mu_heal, self.sigma_heal, self.rv_heal = self.estimate_healthy_pdf(self.data, self.mask, self.params)
        print '\tliver pdf: mu = ', self.mu_heal, ', sigma = ', self.sigma_heal
        # hypodense pdf ------------
        self.mu_hypo, self.sigma_hypo, self.rv_hypo = self.estimate_outlier_pdf(self.data, self.mask, self.rv_heal, 'hypo', self.params)
        print '\thypodense pdf: mu = ', self.mu_hypo, ', sigma= ', self.sigma_hypo
        # hyperdense pdf ------------
        self.mu_hyper, self.sigma_hyper, self.rv_hyper = self.estimate_outlier_pdf(self.data, self.mask, self.rv_heal, 'hyper', self.params)
        print '\thyperdense pdf: mu = ', self.mu_hyper, ', sigma= ', self.sigma_hyper

        self.models = dict()
        # self.models['mu_heal'] = self.mu_heal
        # self.models['sigma_heal'] = self.sigma_heal
        # self.models['mu_hypo'] = self.mu_hypo
        # self.models['sigma_hypo'] = self.sigma_hypo
        # self.models['mu_hyper'] = self.mu_hyper
        # self.models['sigma_hyper'] = self.sigma_hyper
        self.models['rv_heal'] = self.rv_heal
        self.models['rv_hypo'] = self.rv_hypo
        self.models['rv_hyper'] = self.rv_hyper


    def objects_filtration(self):
        min_area = self.params['min_area']
        max_area = self.params['max_area']
        min_compactness = self.params['min_compactness']

        self.filtered_idxs = np.ones(self.n_objects, dtype=np.bool)

        for i in range(self.n_objects):
            # TODO: test the compactness
            if self.areas[i] < min_area or self.areas[i] > max_area:# or self.comps[i] < min_compactness:
                self.filtered_idxs[i] = False



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


        print 'estimating number of clusters...'
        print '\tcurrently switched off'
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

        # calculating intensity models if necesarry
        if not self.models:
            self.calculate_intensity_models()

        # zooming the data
        if self.params['zoom']:
            print 'zooming data...'
            self.data = self.data_zoom(self.data, self.voxel_size, self.params['working_voxel_size_mm'])
            self.mask = self.data_zoom(self.mask, self.voxel_size, self.params['working_voxel_size_mm'])
        else:
            self.data = tools.resize3D(self.data, self.params['scale'], sliceId=0)
            self.mask = tools.resize3D(self.mask, self.params['scale'], sliceId=0)
        # data = data.astype(np.uint8)

        print 'calculating unary potentials...'
        self.status_bar.showMessage('Calculating unary potentials...')
        # create unaries
        # unaries = data_d
        # # as we convert to int, we need to multipy to get sensible values
        # unaries = (1 * np.dstack([unaries, -unaries]).copy("C")).astype(np.int32)
        self.unaries = beta * self.get_unaries(self.data, self.mask, self.models, self.params)
        n_labels = self.unaries.shape[2]

        print 'calculating pairwise potentials...'
        self.status_bar.showMessage('Calculating pairwise potentials...')
        # create potts pairwise
        self.pairwise = -alpha * np.eye(n_labels, dtype=np.int32)

        print 'deriving graph edges...'
        self.status_bar.showMessage('Deriving graph edges...')
        # use the gerneral graph algorithm
        # first, we construct the grid graph
        inds = np.arange(self.data.size).reshape(self.data.shape)
        if self.data.ndim == 2:
            horz = np.c_[inds[:, :-1].ravel(), inds[:, 1:].ravel()]
            vert = np.c_[inds[:-1, :].ravel(), inds[1:, :].ravel()]
            self.edges = np.vstack([horz, vert]).astype(np.int32)
        elif self.data.ndim == 3:
            # horz = np.c_[inds[:, :-1].ravel(), inds[:, 1:].ravel()]
            # vert = np.c_[inds[:-1, :].ravel(), inds[1:, :].ravel()]
            horz = np.c_[inds[:, :, :-1].ravel(), inds[:, :, 1:].ravel()]
            vert = np.c_[inds[:, :-1, :].ravel(), inds[:, 1:, :].ravel()]
            dept = np.c_[inds[:-1, :, :].ravel(), inds[1:, :, :].ravel()]
            self.edges = np.vstack([horz, vert, dept]).astype(np.int32)
        # deleting edges with nodes outside the mask
        nodes_in = np.ravel_multi_index(np.nonzero(self.mask), self.data.shape)
        rows_inds = np.in1d(self.edges, nodes_in).reshape(self.edges.shape).sum(axis=1) == 2
        self.edges = self.edges[rows_inds, :]

        # plt.show()

        # un = unaries[:, :, 0].reshape(data_o.shape)
        # un = unaries[:, :, 1].reshape(data_o.shape)
        # un = unaries[:, :, 2].reshape(data_o.shape)
        # py3DSeedEditor.py3DSeedEditor(un).show()

        print 'calculating graph cut...'
        self.status_bar.showMessage('Calculating graph cut...')
        # we flatten the unaries
        # result_graph = pygco.cut_from_graph(edges, unaries.reshape(-1, 2), pairwise)
        # print 'tu: ', unaries.reshape(-1, n_labels).shape
        result_graph = pygco.cut_from_graph(self.edges, self.unaries.reshape(-1, n_labels), self.pairwise)
        self.res = result_graph.reshape(self.data.shape)

        self.res = np.where(self.mask, self.res, -1)

        # zooming to the original size
        if self.params['zoom']:
            self.res = self.zoom_to_shape(self.res, self.orig_shape)
        else:
            self.res = tools.resize3D(self.res, 1. / self.params['scale'], sliceId=0)

        print '\t...done'
        self.status_bar.showMessage('Done')

        # debug visualization
        # self.viewer = Viewer_3D.Viewer_3D(self.res, range=True)
        # self.viewer.show()


        self.status_bar.showMessage('Extracting objects...')
        labels_hypo, n_hypo = scindimea.label(self.res == hypo_lab)
        labels_hypo -= 1  # shifts background to -1
        labels_hyper, n_hyper = scindimea.label(self.res == hyper_lab)
        labels_hyper -= 1  # shifts background to -1
        self.n_objects = n_hypo + n_hyper
        # self.objects = labels_hypo + (labels_hyper + n_hypo)
        self.objects = labels_hypo
        self.objects = np.where(labels_hyper >= 0, labels_hyper + n_hypo, self.objects)
        self.status_bar.showMessage('Done')

        self.status_bar.showMessage('Calculating object features...')
        print 'Calculating object features...'
        self.areas = self.get_areas(self.objects)
        # self.comps = self.get_compactness(self.objects)
        self.comps = np.zeros(self.n_objects)
        self.labels = np.unique(self.objects)[1:]  # from 1 because the first idex is background (-1)
        # self.features = np.hstack((self.areas, self.comps))

        print 'Initial filtration...'
        self.objects_filtration()

        # self.fill_table(self.areas, self.comps)
        print 'Done'
        self.status_bar.showMessage('Done')

        # if self.params['filtration']:
        #     print 'calculating features of hypodense tumors...'
        #     labels_hypo, n_labels = scindimea.label(self.res == hypo_lab)
        #     labels_hypo -= 1  # shifts background to -1
        #     areas_hypo = np.zeros(n_labels)
        #     comps_hypo = self.get_compactness(labels_hypo)
        #     for i in range(n_labels):
        #         lab = labels_hypo == (i + 1)
        #         areas_hypo[i] = lab.sum()
        #         print 'label = %i, area = %i, comp = %.2f' % (i, areas_hypo[i], comps_hypo[i])
        #         # py3DSeedEditor.py3DSeedEditor(data_o, contour=lab).show()
        #     print '\t...done'
        #
        #     print 'calculating features of hyperdense tumors...'
        #     labels_hyper, n_labels = scindimea.label(self.res == hyper_lab)
        #     labels_hyper -= 1  # shifts background to -1
        #     areas_hyper = np.zeros(n_labels)
        #     comps_hyper = self.get_compactness(labels_hyper)
        #     for i in range(n_labels):
        #         lab = labels_hyper == (i + 1)
        #         areas_hyper[i] = lab.sum()
        #         print 'label = %i, area = %i, comp = %.2f' % (i, areas_hyper[i], comps_hyper[i])
        #         # py3DSeedEditor.py3DSeedEditor(data_o, contour=lab).show()
        #     print '\t...done'
        #
        #     print 'filtering false objects...'
        #     features = ('area', 'compactness')
        #     features_hypo_v = np.vstack((areas_hypo, comps_hypo)).T
        #     features_hyper_v = np.vstack((areas_hyper, comps_hyper)).T
        #     hypo_ok = self.filter_objects(features_hypo_v, features, self.params).sum(axis=1) == len(features)
        #     hyper_ok = self.filter_objects(features_hyper_v, features, self.params).sum(axis=1) == len(features)
        #     print '\tfiltrated hypodense: %i/%i' % (hypo_ok.sum(), hypo_ok.shape[0])
        #     print '\tfiltrated hyperdense: %i/%i' % (hyper_ok.sum(), hyper_ok.shape[0])


        #     TumorVisualiser.run(self.data, self.res, self.params['healthy_label'], self.params['hypo_label'], self.params['hyper_label'], slice_axis=0)

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