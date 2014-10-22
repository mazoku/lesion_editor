__author__ = 'tomas'

import numpy as np
import Computational_core as gc

import scipy.ndimage.measurements as scindimea
import skimage.transform as skitra
from Computational_core import *


params = dict()
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
params['healthy_label'] = 1  # label of hyalthy parenchyma
params['min_area'] = 20
params['min_compactness'] = 0.2

data = np.load('input_orig_data.npy')
data_sm = np.load('input_data.npy')
mask = np.load('mask.npy')

params['slice_idx'] = 33

# plt.figure()
# plt.subplot(221), plt.imshow(data[params['slice_idx'], :, :], 'gray'), plt.title('input_orig_data')
# plt.subplot(222), plt.imshow(data_sm[params['slice_idx'], :, :], 'gray'), plt.title('input_data')
# plt.subplot(224), plt.imshow(mask[params['slice_idx'], :, :], 'gray'), plt.title('mask')
# plt.show()

data = data[params['slice_idx'], :, :]
data_sm = data_sm[params['slice_idx'], :, :]
data_sm = skitra.resize(data_sm, data.shape)
mask = mask[params['slice_idx'], :, :]

# plt.figure()
# plt.subplot(121), plt.imshow(data, 'gray')
# plt.subplot(122), plt.imshow(mask, 'gray')
# plt.show()

# healthy pdf ------------
print 'estimating pdf of healthy parenchym...'
mu_h, sigma_h, rv_healthy = estimate_healthy_pdf(data, mask, params)
print 'liver pdf: mu = ', mu_h, ', sigma = ', sigma_h
mu_h_sm, sigma_h_sm, rv_healthy_sm = estimate_healthy_pdf(data_sm, mask, params)
print 'liver pdf: mu_sm = ', mu_h_sm, ', sigma_sm = ', sigma_h_sm


if data.ndim == 3:
    mask_e = tools.eroding3D(mask, skimor.disk(5))
else:
    mask_e = skimor.binary_erosion(mask, np.ones((5, 5)))
# mask_e = mask
unaries_healthy = - rv_healthy.logpdf(data) * mask_e
prob_healthy = rv_healthy.pdf(data) * mask_e

unaries_healthy_sm = - rv_healthy_sm.logpdf(data_sm) * mask_e
prob_healthy_sm = rv_healthy_sm.pdf(data_sm) * mask_e

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

# PDF prob
prob_hyper_pdf = rv_hypo.pdf(data) * mask_e
prob_hypo_pdf = rv_hyper.pdf(data) * mask_e

# cumsum
x = np.arange(0, 255, 0.1)
hyper_cdf = rv_hyper.cdf(x) * rv_healthy.pdf(mu_h)
hypo_cdf = rv_hypo.cdf(x) * rv_healthy.pdf(mu_h)
hypo_cdf = hypo_cdf.max() - hypo_cdf

prob_hyper_cdf = rv_hyper.cdf(data) * rv_healthy.pdf(mu_h) * mask_e
prob_hypo_cdf = (1 - rv_hypo.cdf(data)) * rv_healthy.pdf(mu_h) * mask_e

# plt.figure()
# plt.plot(x, rv_healthy.pdf(x), 'g')
# plt.hold(True)
# plt.plot(x, hypo_cdf, 'b')
# plt.plot(x, hyper_cdf, 'r')
# plt.show()

# PDF versus CDF --------------------------------------------------------------------------------
plt.figure()
plt.subplot(321), plt.imshow(data, 'gray', interpolation='nearest'), plt.title('input')
plt.subplot(322), plt.imshow(prob_healthy, 'gray', interpolation='nearest'), plt.title('healthy')

plt.subplot(323), plt.imshow(prob_hyper_pdf, 'gray', interpolation='nearest'), plt.title('hypodense - pdf')
plt.subplot(324), plt.imshow(prob_hypo_pdf, 'gray', interpolation='nearest'), plt.title('hyperdense - pdf')

plt.subplot(325), plt.imshow(prob_hypo_cdf, 'gray', interpolation='nearest'), plt.title('hypodense - cdf')
plt.subplot(326), plt.imshow(prob_hyper_cdf, 'gray', interpolation='nearest'), plt.title('hyperdense - cdf')


# ploting GMM ------------------------------------
ints = data[np.nonzero(mask)]
hist, bins = skiexp.histogram(ints, nbins=256)
plt.figure()
plt.subplot(311)
plt.title('histogram of input data')
plt.plot(bins, hist)
ax = plt.axis()
plt.axis([0, 256, ax[2], ax[3]])

plt.subplot(312)
plt.title('GMM')
plt.plot(x, rv_hypo.pdf(x), 'b')
plt.plot(x, rv_healthy.pdf(x), 'g')
plt.plot(x, rv_hyper.pdf(x), 'r')
ax = plt.axis()
plt.axis([0, 256, ax[2], ax[3]])

x = np.arange(0, 255, 0.1)
healthy = rv_healthy.pdf(x)
hypo = (1 - rv_hypo.cdf(x)) * rv_healthy.pdf(mu_h)
hyper = rv_hyper.cdf(x) * rv_healthy.pdf(mu_h)

plt.subplot(313)
plt.title('cdf')
plt.plot(x, healthy, 'g')
plt.plot(x, hypo, 'b')
plt.plot(x, hyper, 'r')
ax = plt.axis()
plt.axis([0, 255, ax[2], ax[3]])


# plt.figure()
# plt.plot(x, healthy, 'g')
# plt.plot(x, hypo, 'b')
# plt.plot(x, hyper, 'r')
# ax = plt.axis()
# plt.axis([0, 255, ax[2], ax[3]])

plt.show()