__author__ = 'tomas'

import numpy as np
import tools

PRIORITY_LOW = 0  # for lessions that are extracted autonomously
PRIORITY_HIGH = 1  # for lessions that are added by the user, these wil not be filtrated by sliders (area, density, ...)

def create_lesion_from_pt(pt, density, lbl, priority=PRIORITY_HIGH):
    """

    :param center: center of lesion, [s, x, y] = [s, c, r]
    :param density:
    :param lbl:
    :return:
    """
    les = Lesion(lbl)

    les.area = 1
    les.compactness = 1
    les.center = pt
    les.priority = priority
    les.mean_density = density
    les.std_density = 0

    les.max_width = 1
    les.max_height = 1
    les.max_depth = 1
    # minimal and maximal row, column and slice
    les.r_min = pt[1]
    les.r_max = pt[1]
    les.c_min = pt[2]
    les.c_max = pt[2]
    les.s_min = pt[0]
    les.s_max = pt[0]

    les.hist = None  # histogram of density

    les.chord = 1  # longest chord (tetiva in czech)

    return les

class Lesion(object):
    """ This class represents lesions. """

    def __init__(self, label, mask=None, data=None, priority=PRIORITY_LOW):
        self.label = label  # label of the lesion in segmented data; its identifier

        self.area = None  # area of the lesion

        self.compactness = None

        self.center = None  # center of mass

        self.priority = priority

        self.mean_density = None
        self.std_density = None

        self.max_width = None
        self.max_height = None
        self.max_depth = None

        # minimal and maximal row, column and slice
        self.r_min = None
        self.r_max = None
        self.c_min = None
        self.c_max = None
        self.s_min = None
        self.s_max = None

        self.hist = None  # histogram of density

        self.chord = None  # longest chord (tetiva in czech)

        if mask is not None:
            self.compute_features(mask, data)

    def compute_features(self, mask, data):
        # getting unique labels that are greater than 0 (0 = background, -1 = out of mask)
        self.area = mask.sum()

        s, r, c = np.nonzero(mask)

        self.compactness = tools.get_zunics_compatness(mask)

        self.center = (s.mean(), r.mean(), c.mean())

        if data is not None:
            pts = data[np.nonzero(mask)]
            self.mean_density = pts.mean()
            self.mean_density_std = pts.std()

        self.r_min = r.min()
        self.r_max = r.max()
        self.c_min = c.min()
        self.c_max = c.max()
        self.s_min = s.min()
        self.s_max = s.max()

        self.max_width = self.c_max - self.c_min
        self.max_height = self.r_max - self.r_min
        self.max_depth = self.c_max - self.c_min

    def __str__(self):
        return 'label=%i, area=%i, mean_dens=%.2f, mean_dens_std=%.2f, center=[%.1f, %.1f, %.1f]' % (
            self.label, self.area, self.mean_density, self.mean_density_std, self.center[0], self.center[1], self.center[2])

def extract_lesions(labels, data=None):
    """
    For each label in 'labels' it creates an instance. Returns list of lesions.
    :param labels: labeled data, lesions have label > 0 (0 = background, -1 = points outside a mask)
    :return: list of lesions
    """
    lesions = list()
    lb_list = [x for x in np.unique(labels) if x > 0]

    for i in lb_list:
        im = labels == i
        lesion = Lesion(i, mask=im, data=data)
        lesions.append(lesion)

    return lesions


if __name__ == '__main__':
    labels = np.array([[1, 1, 0, 2, 0],
                       [1, 1, 0, 2, 0],
                       [0, 0, 0, 2, 0],
                       [3, 0, 4, 0, 5],
                       [3, 0, 4, 0, 0]], dtype=np.int)
    labels = np.dstack((labels, labels, labels))

    lesions = extract_lesions(labels, data=labels)

    for i in lesions:
        print i

# TODO: dopocitat compactness
# TODO: dopocitat chord
# TODO: dopocitat hist