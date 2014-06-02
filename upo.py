__author__ = 'tomas'

import scipy.stats as scista
import numpy as np
import matplotlib.pyplot as plt
import tools
import skimage.exposure as skiexp
import skimage.morphology as skimor

slice_idx = 33
# data = np.load('smoothed_data.npy')
o_data = np.load('input_orig_data.npy')
mask = np.load('mask.npy')
n_slices = mask.shape[0]


# o_data_s = o_data[slice_idx, :, :]
# mask_s = mask[slice_idx, :, :]

# ints = o_data_s[np.nonzero(mask_s)].astype(np.int)
# hist, bins = skiexp.histogram(ints, nbins=256)

ints_o = o_data[np.nonzero(mask)].astype(np.int)
hist_o, bins_o = skiexp.histogram(ints_o, nbins=256)

mu, sigma = scista.norm.fit(ints_o)
rv_h = scista.norm(mu, sigma)
# rv2 = scista.norm(mu, sigma / 2)

# x = np.linspace(rv.ppf(0.01), rv.ppf(0.99), 100)
# plt.figure()
# plt.subplot(211), plt.plot(bins_o, hist_o)
# ax = plt.axis()
# plt.axis([0, 256, ax[2], ax[3]])
# plt.subplot(212), plt.plot(x, rv.pdf(x), 'r-')
# ax = plt.axis()
# plt.axis([0, 256, ax[2], ax[3]])
# plt.show()

# mask_e = tools.eroding3D(mask, skimor.disk(5))
mask_e = mask
probs = rv_h.pdf(o_data) * mask_e

max_prob = rv_h.pdf(mu)
print 'max_prob = %.3f' % max_prob
prob_t = max_prob / 2

ints_out_m = probs < prob_t * mask_e
ints_out = o_data[np.nonzero(ints_out_m)]
hist_out, bins_out = skiexp.histogram(ints_out, nbins=256)

ints_hypo = ints_out[np.nonzero(ints_out < mu)]
ints_hyper = ints_out[np.nonzero(ints_out > mu)]

g = scista.dgamma.fit(ints_out)
rv_outs = scista.dgamma(g[0], loc=g[1], scale=g[2])
x = np.arange(0, 256, 0.1)

g = scista.norm.fit(ints_hypo)
rv_hypo = scista.norm(g[0], g[1])

g = scista.norm.fit(ints_hyper)
rv_hyper = scista.norm(g[0], g[1])

plt.figure()
plt.subplot(211)
plt.plot(bins_out, hist_out)
plt.title('histogram of outliers')
ax = plt.axis()
plt.axis([0, 256, ax[2], ax[3]])
plt.subplot(212)
plt.plot(x, rv_outs.pdf(x), 'm')
plt.title('pdfs')
plt.hold(True)
plt.plot(x, rv_h.pdf(x), 'b')
plt.plot(x, rv_hypo.pdf(x), 'g')
plt.plot(x, rv_hyper.pdf(x), 'r')
ax = plt.axis()
plt.axis([0, 256, ax[2], ax[3]])
plt.legend(['ouliers as dgamma', 'healthy as norm', 'hypo as norm', 'hyper as norm'])
plt.show()

probs_l = rv_h.pdf(o_data)
probs_o = rv_outs.pdf(o_data)
probs_hypo = rv_hypo.pdf(o_data)
probs_hyper = rv_hyper.pdf(o_data)
plt.figure()
plt.subplot(321), plt.imshow(o_data[slice_idx, :, :], 'gray')
plt.subplot(323), plt.imshow(probs_l[slice_idx, :, :], 'gray'), plt.title('helthy as norm')
plt.subplot(324), plt.imshow(probs_o[slice_idx, :, :], 'gray'), plt.title('outliers as dgamma')
plt.subplot(325), plt.imshow(probs_hypo[slice_idx, :, :], 'gray'), plt.title('hypo as norm')
plt.subplot(326), plt.imshow(probs_hyper[slice_idx, :, :], 'gray'), plt.title('hyper as norm')
plt.show()

# h_hyper, b_hyper = skiexp.histogram(ints_hyper, nbins=256)
# h_hypo, b_hypo = skiexp.histogram(ints_hypo, nbins=256)
# ints_hypo_res = skiexp.rescale_intensity(ints_hypo, (ints_hypo.min(), ints_hypo.max()), (ints_hypo.max(), ints_hypo.min()))
# h_hypo_res, b_hypo_res = skiexp.histogram(ints_hypo_res, nbins=256)

# plt.figure()
# plt.plot(b_hyper, h_hyper, 'b')
# plt.hold(True)
# plt.plot(b_hypo, h_hypo, 'm')
# plt.plot(b_hypo_res, h_hypo_res, 'g')
# plt.show()

# shape_hyper, loc_hyper, scale_hyper = scista.lognorm.fit(ints_hyper)
# rv_hyper = scista.lognorm(shape_hyper, loc=loc_hyper, scale=scale_hyper)
# x = np.linspace(rv_hyper.ppf(0.001), rv_hyper.ppf(0.999), 300)

# ints_hypo_res = skiexp.rescale_intensity(ints_hypo, (ints_hypo.min(), ints_hypo.max()), (ints_hypo.max(), ints_hypo.min()))
# shape_hypo, loc_hypo, scale_hypo = scista.lognorm.fit(ints_hypo_res)

plt.figure()
plt.plot(x, rv_hypo.pdf(x))
plt.show()

# plt.figure()
# plt.subplot(211), plt.plot(x, rv_hyper.pdf(x)), plt.title('hyperdense pdf')
# plt.subplot(212), plt.plot(x, rv_hypo.pdf(x)), plt.title('hypodense pdf')
# plt.show()

# plt.figure()
# plt.plot(x, rv_hyper.pdf(x))
# plt.show()

# plt.figure()
# plt.subplot(221), plt.imshow(o_data[slice_idx, :, :], 'gray')
# plt.subplot(222), plt.imshow(probs[slice_idx, :, :], 'gray'), plt.colorbar()
# plt.subplot(223), plt.imshow(probs[slice_idx, :, :] > prob_t, 'gray')
# plt.show()

# for i in range(n_slices):
#     # probs = rv.pdf(o_data[i, :, :]) * mask_e
#     plt.figure()
#     plt.subplot(121), plt.imshow(o_data[i, :, :], 'gray')
#     plt.subplot(122), plt.imshow(probs[i, :, :], 'gray'), plt.colorbar()
#     plt.show()

# HISTOGRAM PO REZECH -----------------------------------------------------------------
# min_size = 100
#
# for i in range(n_slices):
#     print 'slice #%i/%i...' % (i, n_slices)
#     data_s = o_data[i, :, :]
#     mask_s = mask[i, :, :]
#
#     ints = data_s[np.nonzero(mask_s)].astype(np.int)
#     print '\tsize: ', ints.size
#     if ints.size > min_size:
#         hist, bins = skiexp.histogram(ints, nbins=256)
#
#         data_bbox, _ = tools.crop_to_bbox(data_s, mask_s)
#         plt.figure()
#         plt.subplot(211), plt.imshow(data_bbox, 'gray', interpolation='nearest')
#         plt.subplot(212), plt.plot(bins, hist), plt.axis([0, 256, 0, hist.max()])
#         plt.show()
#     else:
#         print '\tto small to estimate anything'
#--------------------------------------------------------------------------------------

# mu =
# sigma =
# rv = scista.lognorm