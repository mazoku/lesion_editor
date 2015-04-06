__author__ = 'tomas'

import numpy as np
import tools
import matplotlib.pyplot as plt
from skimage import graph, data, io, color
import skimage.segmentation as skiseg

slice_idx = 60
# fname = '/home/tomas/Data/liver_segmentation_06mm/tryba/data_other/org-exp_186_49290986_arterial_0.6_B20f-.pklz'

# im, mask, voxel_size = tools.load_pickle_data(fname, slice_idx=slice_idx)

fname = '/home/tomas/Dropbox/images/medicine/hypo2in.png'
im = io.imread(fname, as_grey=True)

labels = skiseg.slic(np.dstack((im, im, im)), compactness=10, n_segments=100)
g = graph.rag_mean_color(im, labels, mode='similarity')
nc = graph.cut_normalized(labels, g)
out2 = color.label2rgb(nc, im, kind='avg')

plt.figure()
plt.subplot(121), plt.imshow(nc)
plt.subplot(122), plt.imshow(out2)
# io.show()

plt.figure()
plt.imshow(im, 'gray')
plt.show()