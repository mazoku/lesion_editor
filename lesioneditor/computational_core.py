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


def data_zoom(data, voxelsize_mm, working_voxelsize_mm):
    zoom = voxelsize_mm / (1.0 * working_voxelsize_mm)
    data_res = scindi.zoom(data, zoom, mode='nearest', order=1).astype(np.int16)
    return data_res

def zoom_to_shape(data, shape):
    zoom = np.array(shape, dtype=np.float) / np.array(data.shape, dtype=np.float)
    data_res = scindi.zoom(data, zoom, mode='nearest', order=1).astype(np.int16)
    return data_res

def estimate_healthy_pdf(data, mask, params):
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

def estimate_outlier_pdf(data, mask, rv_healthy, outlier_type, params):
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

def get_unaries(data, mask, models, params):
    rv_heal = models['heal']
    rv_hyper = models['hyper']
    rv_hypo = models['hypo']
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
    if params['show_unaries']:
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

def get_compactness(labels):
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

def get_areas(labels):
    nlabels = labels.max() + 1
    areas = np.zeros(nlabels)
    for i in range(nlabels):
        obj = labels == i
        areas[i] = obj.sum()

    return areas

def smooth_data(data, params):
    if data is not None:
        # smoothing data
        print 'smoothing data...'
        if params['smoothing'] == 1:
            data = skifil.gaussian_filter(data, sigma=params['sigma'])
        elif params['smoothing'] == 2:
            data = tools.smoothing_bilateral(data, sigma_space=params['sigma_spatial'], sigma_color=params['sigma_range'], sliceId=0)
        elif params['smoothing'] == 3:
            data = tools.smoothing_tv(data, weight=params['weight'], sliceId=0)
        else:
            print '\tcurrently switched off'

    return data

def calculate_intensity_models(data, mask, params):
    print 'calculating intensity models...'
    # liver pdf ------------
    rv_heal = estimate_healthy_pdf(data, mask, params)
    print '\tliver pdf: mu = ', rv_heal.mean(), ', sigma = ', rv_heal.std()
    # hypodense pdf ------------
    rv_hypo = estimate_outlier_pdf(data, mask, rv_heal, 'hypo', params)
    print '\thypodense pdf: mu = ', rv_hypo.mean(), ', sigma = ', rv_hypo.std()
    # hyperdense pdf ------------
    rv_hyper = estimate_outlier_pdf(data, mask, rv_heal, 'hyper', params)
    print '\thyperdense pdf: mu = ', rv_hyper.mean(), ', sigma = ', rv_hyper.std()

    models = dict()
    models['heal'] = rv_heal
    models['hypo'] = rv_hypo
    models['hyper'] = rv_hyper

    return models

def objects_filtration(data, params, selected_labels=None, area=None, density=None, compactness=None):
    if data.lesions is not None:
        lesions = data.lesions[:]  # copy of the list
    else:
        return

    if area is None:
        area = [params['min_area'], params['max_area']]
    if density is None:
        density = [params['min_density'], params['max_density']]
    if compactness is None:
        compactness = params['min_compactness']

    if area is not None:
        lesions = [x for x in lesions if area[0] <= x.area <= area[1] or x.priority == Lesion.PRIORITY_HIGH]

    if density is not None:
        lesions = [x for x in lesions if density[0] <= x.mean_density <= density[1] or x.priority == Lesion.PRIORITY_HIGH]

    if compactness is not None:
        if compactness > 1:
            compactness = float(compactness) / params['compactness_step']
        lesions = [x for x in lesions if compactness <= x.compactness or x.priority == Lesion.PRIORITY_HIGH]

    # geting labels of filtered objects
    filtered_lbls = [x.label for x in lesions]

    if selected_labels is not None:
        filtered_lbls = np.intersect1d(filtered_lbls, selected_labels)

    labels_filt_tmp = np.where(data.labels > params['bgd_label'], params['healthy_label'], params['bgd_label'])
    is_in = np.in1d(data.objects, filtered_lbls).reshape(data.labels.shape)
    data.labels_filt = np.where(is_in, data.labels, labels_filt_tmp)
    # print 'setting lesion_filt ...',
    data.lesions_filt = [x for x in data.lesions if x.label in filtered_lbls]
    # print 'done'

    return filtered_lbls

def run_mrf(data_o, params):
    # slice_idx = self.params['slice_idx']
    alpha = params['alpha']
    beta = params['beta']
    # hypo_lab = params['hypo_label']
    # hyper_lab = params['hyper_label']

    # zooming the data
    print 'rescaling data ...',
    if params['zoom']:
        data = data_zoom(data_o.data, data_o.voxel_size, params['working_voxel_size_mm'])
        mask = data_zoom(data_o.mask, data_o.voxel_size, params['working_voxel_size_mm'])
    else:
        data = tools.resize3D(data_o.data, params['scale'], sliceId=0)
        mask = tools.resize3D(data_o.mask, params['scale'], sliceId=0)
    print 'ok'
    # data = data.astype(np.uint8)

    # calculating intensity models if necesarry
    print 'estimating color models ...'
    if data_o.models is None:
        data_o.models = calculate_intensity_models(data, mask, params)
        print 'ok'
    else:
        print 'already done'

    print 'calculating unary potentials ...',
    # self.status_bar.showMessage('Calculating unary potentials...')
    # create unaries
    # # as we convert to int, we need to multipy to get sensible values
    # unaries = (1 * np.dstack([unaries, -unaries]).copy("C")).astype(np.int32)
    unaries = beta * get_unaries(data, mask, data_o.models, params)
    n_labels = unaries.shape[2]
    print 'ok'

    print 'calculating pairwise potentials ...',
    # self.status_bar.showMessage('Calculating pairwise potentials...')
    # create potts pairwise
    pairwise = -alpha * np.eye(n_labels, dtype=np.int32)
    print 'ok'

    print 'deriving graph edges ...',
    # self.status_bar.showMessage('Deriving graph edges...')
    # use the gerneral graph algorithm
    # first, we construct the grid graph
    inds = np.arange(data.size).reshape(data.shape)
    if data.ndim == 2:
        horz = np.c_[inds[:, :-1].ravel(), inds[:, 1:].ravel()]
        vert = np.c_[inds[:-1, :].ravel(), inds[1:, :].ravel()]
        edges = np.vstack([horz, vert]).astype(np.int32)
    elif data.ndim == 3:
        horz = np.c_[inds[:, :, :-1].ravel(), inds[:, :, 1:].ravel()]
        vert = np.c_[inds[:, :-1, :].ravel(), inds[:, 1:, :].ravel()]
        dept = np.c_[inds[:-1, :, :].ravel(), inds[1:, :, :].ravel()]
        edges = np.vstack([horz, vert, dept]).astype(np.int32)
    # deleting edges with nodes outside the mask
    nodes_in = np.ravel_multi_index(np.nonzero(mask), data.shape)
    rows_inds = np.in1d(edges, nodes_in).reshape(edges.shape).sum(axis=1) == 2
    edges = edges[rows_inds, :]
    print 'ok'

    # plt.show()

    # un = unaries[:, :, 0].reshape(data_o.shape)
    # un = unaries[:, :, 1].reshape(data_o.shape)
    # un = unaries[:, :, 2].reshape(data_o.shape)
    # py3DSeedEditor.py3DSeedEditor(un).show()

    print 'calculating graph cut ...',
    # self.status_bar.showMessage('Calculating graph cut...')
    result_graph = pygco.cut_from_graph(edges, unaries.reshape(-1, n_labels), pairwise)
    labels = result_graph.reshape(data.shape) + 1  # +1 to shift the first class to label number 1

    labels = np.where(mask, labels, params['bgd_label'])

    # zooming to the original size
    if params['zoom']:
        data_o.labels = zoom_to_shape(labels, data_o.orig_shape)
    else:
        data_o.labels = tools.resize3D(labels, shape=data_o.orig_shape, sliceId=0)#.astype(np.int64)

    print 'ok'
    # self.status_bar.showMessage('Done')

    # debug visualization
    # self.viewer = Viewer_3D.Viewer_3D(self.res, range=True)
    # self.viewer.show()

    print 'extracting objects ...',
    # self.status_bar.showMessage('Extracting objects ...'),
    labels_tmp = np.where(data_o.labels == params['healthy_label'], params['bgd_label'], data_o.labels)  # because we can set only one label as bgd
    data_o.objects = skimea.label(labels_tmp, background=params['bgd_label'])
    data_o.lesions = Lesion.extract_lesions(data_o.objects, data_o.data)
    # self.status_bar.showMessage('Done')
    print 'ok'

    # TumorVisualiser.run(self.data, self.res, self.params['healthy_label'], self.params['hypo_label'], self.params['hyper_label'], slice_axis=0)

    # mayavi_visualization(res)
