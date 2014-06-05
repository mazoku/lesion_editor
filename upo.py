__author__ = 'tomas'

import scipy.stats as scista
import numpy as np
import matplotlib.pyplot as plt
import tools
import skimage.exposure as skiexp
import skimage.morphology as skimor
from pygco import cut_simple, cut_from_graph

x = np.zeros((10, 12, 3))
x[:, :4, 0] = -1
x[:, 4:8, 1] = -1
x[:, 8:, 2] = -1
unaries = x + 1. * np.random.normal(size=x.shape)
x = np.argmin(x, axis=2)
unaries = (unaries * 10).astype(np.int32)
x_thresh = np.argmin(unaries, axis=2)

# potts potential
pairwise_potts = -2 * np.eye(3, dtype=np.int32)
result = cut_simple(unaries, 10 * pairwise_potts)

# potential that penalizes 0-1 and 1-2 less thann 0-2
pairwise_1d = -15 * np.eye(3, dtype=np.int32) - 8
pairwise_1d[-1, 0] = 0
pairwise_1d[0, -1] = 0
print(pairwise_1d)
result_1d = cut_simple(unaries, pairwise_1d)

plt.figure()
plt.subplot(221, title="original")
plt.imshow(x, interpolation="nearest")
plt.subplot(222, title="thresholded unaries")
plt.imshow(x_thresh, interpolation="nearest")
plt.subplot(223, title="potts potentials")
plt.imshow(result, interpolation="nearest")
plt.subplot(224, title="1d topology potentials")
plt.imshow(result_1d, interpolation="nearest")
plt.show()
