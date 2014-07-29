__author__ = 'tomas'

import numpy as np
import matplotlib.pyplot as plt

import skimage.io as skiio
import skimage.transform as skitra
import skimage.exposure as skiexp
import skimage.segmentation as skiseg
from sklearn import mixture

import scipy.stats as scista

from pygco import cut_from_graph

import itertools


fname = '/home/tomas/Dropbox/images/Berkeley_Benchmark/man_with_hat_189080.jpg'
im = skiio.imread(fname, as_grey=True)
sf = 0.5
im = skiexp.rescale_intensity(skitra.rescale(im, sf), (0, 1), (0, 255)).astype(np.int)
pixs = im.flatten()

# estimating number of components
lowest_bic = np.infty
bic = []
max_n_components = 5
n_components_range = range(1, max_n_components + 1)
cv_types = ['spherical']#, 'tied', 'diag', 'full']
for cv_type in cv_types:
    print 'cv_type: ', cv_type
    for n_components in n_components_range:
        print '\tn_components: ', n_components
        # Fit a mixture of Gaussians with EM
        gmm = mixture.GMM(n_components=n_components, covariance_type=cv_type)
        gmm.fit(pixs)
        bic.append(gmm.bic(pixs))
        if bic[-1] < lowest_bic:
            lowest_bic = bic[-1]
            best_gmm = gmm
n_components = n_components_range[np.argmin(bic)]

print 'bic score:', bic
print 'optimal number of components: %i, bic: %.1f' % (n_components, lowest_bic)
print 'means:\n', best_gmm.means_
print 'covars:\n', best_gmm.covars_

labels = best_gmm.predict(pixs)
labels = labels.reshape(im.shape)

# labeling
probs = np.zeros((im.shape[0], im.shape[1], n_components))
for i in range(n_components):
    rv = scista.norm(best_gmm.means_[i], np.sqrt(best_gmm.covars_[i]))
    probs[:, :, i] = rv.pdf(pixs).reshape(im.shape)

    # plt.figure()
    # plt.imshow(probs[:, :, i], 'gray')
    # plt.title('component #%i' % i)
# plt.show()


# MRF
#   unary term
beta = 1
unaries = np.zeros((im.shape[0], im.shape[1], n_components))
for i in range(n_components):
    rv = scista.norm(best_gmm.means_[i], np.sqrt(best_gmm.covars_[i]))
    unaries[:, :, i] = - beta * rv.logpdf(im)
unaries = unaries.astype(np.int32)

for i in range(n_components):
    plt.figure()
    plt.subplot(121), plt.imshow(probs[:, :, i], 'gray'), plt.title('probability for component #%i' % i)
    plt.subplot(122), plt.imshow(unaries[:, :, i], 'gray'), plt.title('unary term for component #%i' % i)
# plt.show()

#   pairwise term
alpha = 50
pairwise = - alpha * np.eye(n_components, dtype=np.int32)

print 'deriving graph edges...'
# use the general graph algorithm
# first, we construct the grid graph
inds = np.arange(im.size).reshape(im.shape)
horz = np.c_[inds[:, :-1].ravel(), inds[:, 1:].ravel()]
vert = np.c_[inds[:-1, :].ravel(), inds[1:, :].ravel()]
edges = np.vstack([horz, vert]).astype(np.int32)

# we flatten the unaries
result_graph = cut_from_graph(edges, unaries.reshape(-1, n_components), pairwise)
res = result_graph.reshape(im.shape)

plt.figure()
plt.subplot(121), plt.imshow(im, 'gray'), plt.title('input image')
plt.subplot(122), plt.imshow(res, 'gray'), plt.title('segmentation')
# plt.show()

# plotting results
color_iter = itertools.cycle(['r', 'g', 'b', 'c', 'm', 'y', 'k'])
x = np.arange(0, 255, 0.1)
hist, bins = skiexp.histogram(pixs, nbins=256)

plt.figure()
plt.subplot(211), plt.plot(bins, hist)
ax = plt.axis()
plt.axis([0, 256, ax[2], ax[3]])
plt.subplot(212)
for i in range(n_components):
    rv = scista.norm(best_gmm.means_[i], np.sqrt(best_gmm.covars_[i]))
    plt.plot(x, rv.pdf(x), color_iter.next())
ax = plt.axis()
plt.axis([0, 256, ax[2], ax[3]])


plt.figure()
plt.subplot(121), plt.imshow(im, 'gray')
plt.subplot(122), plt.imshow(labels, interpolation='nearest')

plt.show()
