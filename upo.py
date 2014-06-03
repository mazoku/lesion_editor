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

ints_o = o_data[np.nonzero(mask)]
hist_o, bins_o = skiexp.histogram(ints_o, nbins=256)

mu, sigma = scista.norm.fit(ints_o)
print mu, sigma
rv_h = scista.norm(mu, sigma)

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

# plt.figure()
# plt.plot(x, rv_hypo.pdf(x), 'b')
# plt.hold(True)
# plt.plot(x, rv_h.pdf(x) + rv_hyper.pdf(x), 'r')
# plt.legend(['hypo as norm', 'healthy + hyper, both as norm'])
# plt.show()

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
#
# probs_l = rv_h.pdf(o_data)
# probs_o = rv_outs.pdf(o_data)
# probs_hypo = rv_hypo.pdf(o_data)
# probs_hyper = rv_hyper.pdf(o_data)
# plt.figure()
# plt.subplot(321), plt.imshow(o_data[slice_idx, :, :], 'gray')
# plt.subplot(323), plt.imshow(probs_l[slice_idx, :, :], 'gray'), plt.title('helthy as norm')
# plt.subplot(324), plt.imshow(probs_o[slice_idx, :, :], 'gray'), plt.title('outliers as dgamma')
# plt.subplot(325), plt.imshow(probs_hypo[slice_idx, :, :], 'gray'), plt.title('hypo as norm')
# plt.subplot(326), plt.imshow(probs_hyper[slice_idx, :, :], 'gray'), plt.title('hyper as norm')
# plt.show()

