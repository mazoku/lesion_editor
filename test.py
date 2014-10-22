__author__ = 'tomas'

import numpy as np
import Computational_core as gc

import scipy.ndimage.measurements as scindimea
from Computational_core import *


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

data_o = np.load('input_orig_data.npy')
res = np.load('res.npy')

hypo_lab = params['hypo_label']
hyper_lab = params['hyper_label']

# print 'calculating features of hypodense tumors...'
# labels_hypo, n_labels = scindimea.label(res == hypo_lab)
# labels_hypo -= 1
# areas_hypo = np.zeros(n_labels)
# comps_hypo = get_compactness(labels_hypo)
# for i in range(n_labels):
#     lab = labels_hypo == i
#     areas_hypo[i] = lab.sum()
#     print 'label = %i, area = %i, comp = %.3f' % (i, areas_hypo[i], comps_hypo[i])
#     # py3DSeedEditor.py3DSeedEditor(data_o, contour=lab).show()
# print '\t...done'

print 'calculating features of hyperdense tumors...'
labels_hyper, n_labels = scindimea.label(res == hyper_lab)
labels_hyper -= 1
areas_hyper = np.zeros(n_labels)
comps_hyper = get_compactness(labels_hyper)
for i in range(n_labels):
    lab = labels_hyper == i
    areas_hyper[i] = lab.sum()
    print 'label = %i, area = %i, comp = %.3f' % (i, areas_hyper[i], comps_hyper[i])
    # py3DSeedEditor.py3DSeedEditor(data_o, contour=lab).show()
print '\t...done'

print 'filtering false objects...'
features = ('area', 'compactness')
# features_hypo_v = np.vstack((areas_hypo, comps_hypo)).T
features_hyper_v = np.vstack((areas_hyper, comps_hyper)).T
# hypo_ok = filter_objects(features_hypo_v, features, params).sum(axis=1) == len(features)
hyper_ok = filter_objects(features_hyper_v, features, params).sum(axis=1) == len(features)

# print '\tfiltrated hypodense: %i/%i' % (hypo_ok.sum(), len(hypo_ok))
print '\tfiltrated hyperdense: %i/%i survived' % (hyper_ok.sum(), len(hyper_ok))


# hypo = np.in1d(labels_hypo, np.nonzero(hypo_ok)).reshape(labels_hypo.shape)
hyper = np.in1d(labels_hyper, np.nonzero(hyper_ok)).reshape(labels_hyper.shape)
py3DSeedEditor.py3DSeedEditor(data_o, contour=hyper).show()