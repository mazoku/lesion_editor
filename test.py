__author__ = 'tomas'

import numpy as np
import graph_cut as gc


res = np.load('res.npy')
gc.mayavi_visualization(res)