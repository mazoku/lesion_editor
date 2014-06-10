# SOMETHING LIKE UNIDENTIFIED PYTHON OBJECT
# ONLY FOR TESTING, SCRIBBLING ETC.

__author__ = 'tomas'


import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage.morphology as scindimor
import scipy.ndimage.measurements as scindimea
import scipy.stats as scista

import skimage.segmentation as skiseg
import skimage.morphology as skimor
import skimage.filter as skifil
import skimage.exposure as skiexp
import skimage.measure as skimea

import cv2
import pygco

import tools
import py3DSeedEditor
from mayavi import mlab

#-------------------------------------------------------------
def load_data(slice_idx=-1):
    # 33 ... hypodense
    # 41 ... small tumor
    # 138 ... hyperdense

    # labels = np.load('label_im.npy')
    data = np.load('input_data.npy')
    o_data = np.load('input_orig_data.npy')
    mask = np.load('mask.npy')

    if slice_idx != -1:
        data_s = data[slice_idx, :, :]
        o_data_s = o_data[slice_idx, :, :]
        mask_s = mask[slice_idx, :, :]

        data, _ = tools.crop_to_bbox(data_s, mask_s)
        o_data, _ = tools.crop_to_bbox(o_data_s, mask_s)
        mask, _ = tools.crop_to_bbox(mask_s, mask_s)

    # plt.figure()
    # plt.subplot(131), plt.imshow(data_bbox, 'gray')
    # plt.subplot(132), plt.imshow(o_data_bbox, 'gray')
    # plt.subplot(133), plt.imshow(mask_bbox, 'gray')
    # plt.show()

    return data, o_data, mask


def estimate_healthy_pdf(data, mask, tparams):
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
        perc_in = n_pts * perc

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
        mu = bins[peak_idx]
        sigma = k_std_l * np.std(inners)

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
    max_prob = rv_healthy.pdf(rv_healthy.mean())
    # print 'max_prob = %.3f' % max_prob
    prob_t = prob_w * max_prob

    ints_out_m = probs < prob_t * mask

    ints_out = data[np.nonzero(ints_out_m)]
    hist, bins = skiexp.histogram(ints_out, nbins=256)

    if outlier_type == 'hypo':
        ints = ints_out[np.nonzero(ints_out < rv_healthy.mean())]
    elif outlier_type == 'hyper':
        ints = ints_out[np.nonzero(ints_out > rv_healthy.mean())]
    else:
        print 'Wrong outlier specification.'
        return

    mu, sigma = scista.norm.fit(ints)

    # hack for moving pdfs of hypo and especially of hyper furhter from the pdf of healthy parenchyma
    mu += hack_mu
    sigma += hack_sigma
    #-----------

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
        # plt.show()

    return mu, sigma, rv


def get_unaries(data, mask, params):
    show_me = params['show_estimated_pdfs']

    # liver pdf ------------
    print 'estimating pdf of healthy parenchym...'
    mu_h, sigma_h, rv_healthy = estimate_healthy_pdf(data, mask, params)
    print 'liver pdf: mu = ', mu_h, ', sigma = ', sigma_h

    # mask_e = skimor.binary_erosion(mask, np.ones((5, 5)))
    # liver_probs = rv_healthy.pdf(data) * mask_e
    # plt.figure()
    # plt.subplot(121), plt.imshow(liver_probs, 'gray'), plt.colorbar()
    # plt.subplot(122), plt.imshow(liver_probs > 0.05, 'gray')
    # plt.show()

    # hypodense pdf ------------
    print 'estimating pdf of hypodense objects...'
    # mu_hypo, sigma_hypo, rv_hypo = estimate_hypo_pdf(data, mask, rv_healthy, show_me)
    mu_hypo, sigma_hypo, rv_hypo = estimate_outlier_pdf(data, mask, rv_healthy, 'hypo', params)
    print 'hypodense pdf: mu = ', mu_hypo, ', sigma= ', sigma_hypo

    # hyperdense pdf ------------
    print 'estimating pdf of hyperdense objects...'
    # mu_hyper, sigma_hyper, rv_hyper = estimate_hyper_pdf(data, mask, rv_healthy, show_me)
    mu_hyper, sigma_hyper, rv_hyper = estimate_outlier_pdf(data, mask, rv_healthy, 'hyper', params)
    print 'hyperdense pdf: mu = ', mu_hyper, ', sigma= ', sigma_hyper

    if show_me:
        ints = data[np.nonzero(mask)]
        hist, bins = skiexp.histogram(ints, nbins=256)
        x = np.arange(0, 255, 0.01)
        plt.figure()
        plt.subplot(211)
        plt.plot(bins, hist)
        ax = plt.axis()
        plt.axis([0, 256, ax[2], ax[3]])
        plt.title('histogram of input data')
        plt.subplot(212)
        plt.plot(x, rv_hypo.pdf(x), 'm')
        plt.hold(True)
        plt.plot(x, rv_healthy.pdf(x), 'b')
        plt.plot(x, rv_hyper.pdf(x), 'g')
        ax = plt.axis()
        plt.axis([0, 256, ax[2], ax[3]])
        plt.title('histogram of input data')
        plt.legend(['hypodense pdf', 'healthy pdf', 'hyperdense pdf'])
        # plt.show()

    # unaries_l = rv_l.pdf(data)
    # unaries_t = rv_t.pdf(data)
    if data.ndim == 3:
        mask_e = tools.eroding3D(mask, skimor.disk(5))
    else:
        mask_e = skimor.binary_erosion(mask, np.ones((5, 5)))
    # mask_e = mask
    # unaries_bcg = - (rv_healthy.logpdf(data) + rv_hyper.logpdf(data)) * mask_e
    unaries_healthy = - rv_healthy.logpdf(data) * mask_e
    unaries_hyper = - rv_hyper.logpdf(data) * mask_e
    unaries_hypo = - rv_hypo.logpdf(data) * mask_e

    # # display unary potentials
    # plt.figure()
    # plt.subplot(221), plt.imshow(data, 'gray')
    # plt.subplot(223), plt.imshow(unaries_l, 'gray'), plt.title('liver unaries'), plt.colorbar()
    # plt.subplot(224), plt.imshow(unaries_t, 'gray'), plt.title('tumor unaries'), plt.colorbar()
    # # plt.show()
    #
    # display estimated pdf of normal distribution
    # ints = data[np.nonzero(mask)]
    # hist, bins = skiexp.histogram(ints, nbins=256)
    # plt.figure()
    # plt.subplot(211)
    # plt.plot(bins, hist)
    # plt.hold(True)
    # plt.plot([mu_h, mu_h], [0, hist.max()], 'g')
    # plt.plot([mu_hypo, mu_hypo], [0, hist.max()], 'm')
    # plt.subplot(212), plt.plot(bins, rv_healthy.pdf(bins), 'g')
    # plt.subplot(212), plt.plot(bins, rv_hypo.pdf(bins), 'm')
    # plt.hold(True)
    # plt.plot(mu_h, rv_healthy.pdf(mu_h), 'go')
    # plt.plot(mu_hypo, rv_hypo.pdf(mu_hypo), 'mo')
    # plt.show()

    # # display histogram with borders of points from which the norm pdf is estimated
    # plt.figure()
    # plt.subplot(211), plt.plot(bins, hist)
    # plt.hold(True)
    # plt.plot(bins[peak_idx], hist[peak_idx], 'ro')
    # plt.plot([bins[peak_idx - win_width], bins[peak_idx - win_width]], [0, hist[peak_idx]], 'r')
    # plt.plot([bins[peak_idx + win_width], bins[peak_idx + win_width]], [0, hist[peak_idx]], 'r')
    # plt.subplot(212), plt.plot(bins, rv.pdf(bins))
    # plt.hold(True)
    # plt.plot(bins[peak_idx], rv.pdf(bins[peak_idx]), 'ro')
    # plt.show()

    # unaries = np.dstack((unaries_hyper, unaries_hypo)).astype(np.int32)
    # np.dstack([tdata1.reshape(-1,1), tdata2.reshape(-1,1)]).copy("C")
    unaries = np.dstack((unaries_hypo.reshape(-1,1), unaries_healthy.reshape(-1,1), unaries_hyper.reshape(-1,1)))
    unaries = unaries.astype(np.int32)
    return unaries


def mayavi_visualization(res):
    ### Read the data in a numpy 3D array ##########################################
    #parenchym = np.logical_or(liver, vessels)
    parenchym = res > 0
    hypodense = res == 0
    hyperdense = res == 2
    data = np.where(parenchym, 1, 0)
    data = data.T

    src = mlab.pipeline.scalar_field(data)
    src.spacing = [1, 1, 1]

    data2 = np.where(hypodense, 1, 0)
    # data2 = hypodense
    data2 = data2.T
    src2 = mlab.pipeline.scalar_field(data2)
    src2.spacing = [1, 1, 1]

    data3 = np.where(hyperdense, 1, 0)
    # data3 = hyperdense
    data3 = data3.T
    src3 = mlab.pipeline.scalar_field(data3)
    src3.spacing = [1, 1, 1]

    #contours 6 ... cevy
    #contours 3 ... jatra
    #contours 10 ... jatra a cevy
    #mlab.pipeline.iso_surface(src, contours=3, opacity=0.1)
    #mlab.pipeline.iso_surface(src, contours=2, opacity=0.2)
    mlab.pipeline.iso_surface(src, contours=2, opacity=0.2, color=(0, 1, 0))
    mlab.pipeline.iso_surface(src2, contours=2, opacity=0.2, color=(1, 0, 0))
    mlab.pipeline.iso_surface(src3, contours=2, opacity=0.2, color=(0, 0, 1))

    mlab.show()


def get_compactness(labels):
    nlabels = labels.max()
    eccs = np.zeros(nlabels)

    for lab in range(nlabels):
        obj = labels == (lab + 1)
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


def filter_objects(feature_v, features, params):
    min_area = params['min_area']
    min_comp = params['min_compactness']

    obj_ok = np.zeros(feature_v.shape, dtype=np.bool)
    area_idx = -1
    comp_idx = -1
    try:
        area_idx = features.index('area')
    except ValueError:
        pass
    try:
        comp_idx = features.index('compactness')
    except ValueError:
        pass

    if area_idx != -1:
        obj_ok[:, area_idx] = feature_v[:, area_idx] > min_area

    if comp_idx != -1:
        obj_ok[:, comp_idx] = feature_v[:, comp_idx] > min_comp

    return obj_ok


def run(params, show_me,):
    slice_idx = params['slice_idx']
    alpha = params['alpha']
    beta = params['beta']
    hypo_lab = params['hypo_label']
    hyper_lab = params['hyper_label']

    _, data_o, mask = load_data(slice_idx)
    # data_o = cv2.imread('/home/tomas/Dropbox/images/medicine/hypodense_bad2.png', 0).astype(np.float)

    # data_s = skifil.gaussian_filter(data_o, sigma)
    #
    # #imd = np.absolute(im - ims)
    # data_d = data_o - data_s
    #
    # data_d = np.where(data_d < 0, 0, data_d)

    print 'calculating unary potentials...'
    # create unaries
    # unaries = data_d
    # # as we convert to int, we need to multipy to get sensible values
    # unaries = (1 * np.dstack([unaries, -unaries]).copy("C")).astype(np.int32)
    unaries = beta * get_unaries(data_o, mask, params)
    n_labels = unaries.shape[2]

    print 'calculating pairwise potentials...'
    # create potts pairwise
    pairwise = -alpha * np.eye(n_labels, dtype=np.int32)

    print 'deriving graph edges...'
    # use the gerneral graph algorithm
    # first, we construct the grid graph
    inds = np.arange(data_o.size).reshape(data_o.shape)
    if data_o.ndim == 2:
        horz = np.c_[inds[:, :-1].ravel(), inds[:, 1:].ravel()]
        vert = np.c_[inds[:-1, :].ravel(), inds[1:, :].ravel()]
        edges = np.vstack([horz, vert]).astype(np.int32)
    elif data_o.ndim == 3:
        # horz = np.c_[inds[:, :-1].ravel(), inds[:, 1:].ravel()]
        # vert = np.c_[inds[:-1, :].ravel(), inds[1:, :].ravel()]
        horz = np.c_[inds[:, :, :-1].ravel(), inds[:, :, 1:].ravel()]
        vert = np.c_[inds[:, :-1, :].ravel(), inds[:, 1:, :].ravel()]
        dept = np.c_[inds[:-1, :, :].ravel(), inds[1:, :, :].ravel()]
        edges = np.vstack([horz, vert, dept]).astype(np.int32)
    # deleting edges with nodes outside the mask
    nodes_in = np.ravel_multi_index(np.nonzero(mask), data_o.shape)
    rows_inds = np.in1d(edges, nodes_in).reshape(edges.shape).sum(axis=1) == 2
    edges = edges[rows_inds, :]

    # plt.show()

    # un = unaries[:, :, 0].reshape(data_o.shape)
    # un = unaries[:, :, 1].reshape(data_o.shape)
    # un = unaries[:, :, 2].reshape(data_o.shape)
    # py3DSeedEditor.py3DSeedEditor(un).show()

    print 'calculating graph cut...'
    # we flatten the unaries
    # result_graph = pygco.cut_from_graph(edges, unaries.reshape(-1, 2), pairwise)
    # print 'tu: ', unaries.reshape(-1, n_labels).shape
    result_graph = pygco.cut_from_graph(edges, unaries.reshape(-1, n_labels), pairwise)
    res = result_graph.reshape(data_o.shape)

    res = np.where(mask, res, -1)
    print '\t...done'

    print 'calculating features of hypodense tumors...'
    labels_hypo, n_labels = scindimea.label(res == hypo_lab)
    areas_hypo = np.zeros(n_labels)
    comps_hypo = get_compactness(labels_hypo)
    for i in range(n_labels):
        lab = labels_hypo == (i + 1)
        areas_hypo[i] = lab.sum()
        print 'label = %i, area = %i, comp = %.2f' % (i, areas_hypo[i], comps_hypo[i])
        # py3DSeedEditor.py3DSeedEditor(data_o, contour=lab).show()
    print '\t...done'

    print 'calculating features of hyperdense tumors...'
    labels_hyper, n_labels = scindimea.label(res == hyper_lab)
    areas_hyper = np.zeros(n_labels)
    comps_hyper = get_compactness(labels_hyper)
    for i in range(n_labels):
        lab = labels_hyper == (i + 1)
        areas_hyper[i] = lab.sum()
        print 'label = %i, area = %i, comp = %.2f' % (i, areas_hyper[i], comps_hyper[i])
        # py3DSeedEditor.py3DSeedEditor(data_o, contour=lab).show()
    print '\t...done'

    print 'filtering false objects...'
    features = ('area', 'compactness')
    features_hypo_v = np.vstack((areas_hypo, comps_hypo)).T
    features_hyper_v = np.vstack((areas_hyper, comps_hyper)).T
    hypo_ok = filter_objects(features_hypo_v, features, params).sum(axis=1) == len(features)
    hyper_ok = filter_objects(features_hyper_v, features, params).sum(axis=1) == len(features)
    print '\tfiltrated hypodense: %i/%i' % (hypo_ok.sum(), hypo_ok.shape[0])
    print '\tfiltrated hyperdense: %i/%i' % (hyper_ok.sum(), hyper_ok.shape[0])

    # print 'calculating compactness...'
    # # labels_uniq = np.unique(res[res > -1])
    # # labels = skimor.label(res)
    # labels = tools.label_3D(res, class_labels=(0, 2))
    # py3DSeedEditor.py3DSeedEditor(labels).show()
    # comps = get_compactness(labels)
    # for i in range(labels.max() + 1):
    #     lab = labels == i
    #     print 'label = %i, comp = %.2f' % (i, comps[i])
    #     py3DSeedEditor.py3DSeedEditor(data_o, contour=lab).show()
    # print '\t...done'

    # np.save('res.npy', res)

    # plt.figure()
    # plt.subplot(131), plt.imshow(data_o, 'gray', vmin=0, vmax=255)
    # plt.subplot(132), plt.imshow(data_d, 'gray')
    # plt.subplot(133), plt.imshow(data_s, 'gray', vmin=0, vmax=255)
    # plt.show()

    if data_o.ndim == 2:
        plt.figure()
        plt.subplot(2, n_labels, 1), plt.title('original')
        plt.imshow(data_o, 'gray', interpolation='nearest')
        plt.subplot(2, n_labels, 2), plt.title('graph cut')
        plt.imshow(res, 'jet', interpolation='nearest', vmin=res.min(), vmax=res.max()), plt.colorbar(ticks=np.unique(res))
        if n_labels == 2:
            k = 3
        else:
            k = 4
        plt.subplot(2, n_labels, k), plt.title('unary labels = 0')
        plt.imshow(unaries[:, :, 0], 'gray', interpolation='nearest'), plt.colorbar()
        plt.subplot(2, n_labels, k + 1), plt.title('unary labels = 1')
        plt.imshow(unaries[:, :, 1], 'gray', interpolation='nearest'), plt.colorbar()
        if n_labels == 3:
            plt.subplot(2, n_labels, k + 2), plt.title('unary labels = 2')
            plt.imshow(unaries[:, :, 2], 'gray', interpolation='nearest'), plt.colorbar()
        plt.show()
    elif data_o.ndim == 3:
        # py3DSeedEditor.py3DSeedEditor(data_o).show()
        # py3DSeedEditor.py3DSeedEditor(res).show()
        mayavi_visualization(res)


#-------------------------------------------------------------
if __name__ == '__main__':
    # 33 ... hypodense
    # 41 ... small tumor
    # 138 ... hyperdense
    # -1 ... all data
    params = dict()
    params['slice_idx'] = -1
    params['sigma'] = 10  # sigma for gaussian blurr
    params['alpha'] = 3  # weightening parameter for pairwise term
    params['beta'] = 1  # weightening parameter for unary term
    params['perc'] = 0.3  # what portion of liver parenchym around peak is used to calculate std of liver normal pdf
    params['k_std_h'] = 3  # weighting parameter for sigma of normal distribution of healthy parenchym
    params['k_std_t'] = 3  # weighting parameter for sigma of normal distribution of tumor
    params['tv_weight'] = 0.05  # weighting parameter for total variation filter
    params['healthy_simple_estim'] = False  # simple healthy parenchym pdf estimation from all data
    params['prob_w'] = 0.5  # prob_w * max_prob is a threshold for data that will be used for estimation of other pdfs

    params['hack_hypo_mu'] = -0  # hard move of mean of hypodense pdf to the left
    params['hack_hypo_sigma'] = 0  # hard widening of sigma of hypodense pdf
    params['hack_hyper_mu'] = 5  # hard move of mean of hyperdense pdf to the right
    params['hack_hyper_sigma'] = 5  # hard widening of sigma of hyperdense  pdf

    params['show_healthy_pdf_estim'] = False
    params['show_estimated_pdfs'] = True
    params['show_outlier_pdf_estim'] = False

    params['hypo_label'] = 0  # label of hypodense objects
    params['hyper_label'] = 2  # label of hyperdense objects
    params['min_area'] = 20
    params['min_compactness'] = 0.2

    show_me = True  # debug visualization

    run(params, show_me)