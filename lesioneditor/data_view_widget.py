__author__ = 'tomas'

import numpy as np

import matplotlib.pyplot as plt

from PyQt4.QtCore import Qt, QSize, QString, SIGNAL
from PyQt4.QtGui import QImage, QDialog,\
    QApplication, QSlider, QPushButton,\
    QLabel, QPixmap, QPainter, qRgba,\
    QComboBox, QIcon, QStatusBar,\
    QHBoxLayout, QVBoxLayout, QFrame, QSizePolicy


# BGRA order
GRAY_COLORTABLE = np.array([[ii, ii, ii, 255] for ii in range(256)],
                           dtype=np.uint8)

SEEDS_COLORTABLE = np.array([[0, 255, 0, 255],
                             [0, 0, 255, 255]], dtype=np.uint8)

LABELS_COLORTABLE = np.array([[0, 0, 0, 255],
                              [255, 0, 0, 255],
                              [0, 255, 0, 255],
                              [0, 0, 255, 255]], dtype=np.uint8)

CONTOURS_COLORS = {
    1: [255, 0, 0],
    2: [0, 0, 255],
    3: [0, 255, 255],
    10: [255, 255, 0],
    11: [0, 255, 0],
    12: [0, 255, 255],
    13: [0, 0, 255],
}

# VIEW_TABLE = {'axial': (2,1,0),
#               'sagittal': (1,0,2),
#               'coronal': (2,0,1)}

# VIEW_TABLE = {'axial': (1,0),
#               'sagittal': (1,0,2),
#               'coronal': (2,0,1)}

CONTOURS_COLORTABLE = np.zeros((256,4), dtype=np.uint8)
CONTOURS_COLORTABLE[:,:3] = 255
CONTOURLINES_COLORTABLE = np.zeros((256,2,4), dtype=np.uint8)
CONTOURLINES_COLORTABLE[:,:,:3] = 255

for ii, jj in CONTOURS_COLORS.iteritems():
    key = ii - 1
    CONTOURS_COLORTABLE[key,:3] = jj
    CONTOURS_COLORTABLE[key,3] = 64
    CONTOURLINES_COLORTABLE[key,0,:3] = jj
    CONTOURLINES_COLORTABLE[key,0,3] = 16
    CONTOURLINES_COLORTABLE[key,1,:3] = jj
    CONTOURLINES_COLORTABLE[key,1,3] = 255


NEI_TAB = [[-1, -1], [0, -1], [1, -1],
           [-1, 0], [1, 0],
           [-1, 1], [0, 1], [1, 1]]


def erase_reg(arr, p, val=0):
    from scipy.ndimage.measurements import label

    labs, num = label(arr)
    aval = labs[p]
    idxs = np.where(labs == aval)
    arr[idxs] = val


class SliceBox(QLabel):

    # def __init__(self, sliceSize, grid, mode='seeds'):
    def __init__(self, sliceSize, mode='seeds'):

        QLabel.__init__(self)

        # self.drawing = False
        # self.modified = False
        # self.last_position = None
        # self.imagesize = QSize(int(sliceSize[0] * grid[0]),
        #                        int(sliceSize[1] * grid[1]))

        self.SHOW_IM = 0  # flag for showing density data
        self.SHOW_LABELS = 1  # flag for showing labels
        self.SHOW_CONTOURS = 2  # flag for showing contours

        self.imagesize = QSize(sliceSize[0], sliceSize[1])
        # self.grid = grid
        self.slice_size = sliceSize
        self.ctslice_rgba = None
        self.cw = {'c': 1.0, 'w': 1.0}

        self.seeds = None
        self.contours = None
        self.contours_old = None
        self.mask_points = None
        self.contour_mode = 'fill'
        # self.contour_mode = 'contours'

        self.show_mode = self.SHOW_IM

        # self.actual_slice = 0
        # self.n_slices = 0
        self.scroll_fun = None

        self.image = QImage(self.imagesize, QImage.Format_RGB32)
        self.setPixmap(QPixmap.fromImage(self.image))
        self.setScaledContents(True)

        self.actual_view = 'axial'
        # self.act_transposition = VIEW_TABLE[self.actual_view]
        self.act_transposition = (1,0)

    def reinit(self, sliceSize):
        self.imagesize = QSize(sliceSize[0], sliceSize[1])
        self.slice_size = sliceSize
        self.ctslice_rgba = None

        self.seeds = None
        self.contours = None
        self.contours_old = None
        self.mask_points = None

        self.image = QImage(self.imagesize, QImage.Format_RGB32)
        self.setPixmap(QPixmap.fromImage(self.image))
        self.setScaledContents(True)

    # def set_data(self, data, type):
    #     self.data = data
    #     self.n_slices, self.n_rows, self.n_cols = data.shape
    #     self.type = type
    #
    #     self.imagesize = QSize(int(self.n_rows * self.grid[0]),
    #                            int(self.n_cols * self.grid[1]))
    #     self.slice_size = (self.n_rows, self.n_cols)

    # def set_slice_size(self, sliceSize):
    #     self.imagesize = QSize(sliceSize[0], sliceSize[1])
    #     self.slice_size = sliceSize
    #
    #     self.seeds = None
    #     self.contours = None
    #     self.contours_old = None
    #     self.mask_points = None
    #
    #     self.image = QImage(self.imagesize, QImage.Format_RGB32)
    #     self.setPixmap(QPixmap.fromImage(self.image))
    #     self.setScaledContents(True)

    def setContours(self, contours):
        self.contours = contours
        self.contours_aview = self.contours.transpose(self.act_transposition)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawImage(event.rect(), self.image)
        painter.end()

    def get_contours(self, img, sl):
        idxs = sl.nonzero()[0]
        keys = np.unique(sl[idxs])
        for ii in keys:
            if ii == 0:
                continue
            aux = np.zeros_like(sl)
            idxsi = np.where(sl == ii)[0]
            aux[idxsi] = 1
            cnt = self.gen_contours(aux)

            self.composeRgba(img, cnt,
                             CONTOURLINES_COLORTABLE[ii - 1,...])

    def gen_contours(self, sl):
        sls = sl.reshape(self.slice_size, order='F')
        cnt = sls.copy()
        chunk = np.zeros((cnt.shape[1] + 2,), dtype=np.int8)
        for irow, row in enumerate(sls):
            chunk[1:-1] = row
            chdiff = np.diff(chunk)
            idx1 = np.where(chdiff > 0)[0]
            if idx1.shape[0] > 0:
                idx2 = np.where(chdiff < 0)[0]
                if idx2.shape[0] > 0:
                    cnt[irow,idx1] = 2
                    cnt[irow,idx2 - 1] = 2

        chunk = np.zeros((cnt.shape[0] + 2,), dtype=np.int8)
        for icol, col in enumerate(sls.T):
            chunk[1:-1] = col
            chdiff = np.diff(chunk)
            idx1 = np.where(chdiff > 0)[0]
            if idx1.shape[0] > 0:
                idx2 = np.where(chdiff < 0)[0]
                if idx2.shape[0] > 0:
                    cnt[idx1,icol] = 2
                    cnt[idx2 - 1,icol] = 2

        return cnt.ravel(order='F')

    def composeRgba(self, bg, fg, cmap):
        idxs = fg.nonzero()[0]

        if idxs.shape[0] > 0:
            fg_rgb = cmap[fg[idxs] - 1]

            af = fg_rgb[...,3].astype(np.uint32)
            rgbf = fg_rgb[...,:3].astype(np.uint32)
            rgbb = bg[idxs,:3].astype(np.uint32)

            rgbx = ((rgbf.T * af).T + (rgbb.T * (255 - af)).T) / 255
            bg[idxs,:3] = rgbx.astype(np.uint8)

    def overRgba(self, bg, fg, cmap):
        idxs = fg.nonzero()[0]
        bg[idxs] = cmap[fg[idxs] - 1]

    def window_slice(self, ctslice):
        if self.win_w > 0:
            mul = 255. / float(self.win_w)
        else:
            mul = 0

        lb =self.win_l - self.win_w / 2
        aux = (ctslice - lb) * mul
        aux = np.where(aux < 0, 0, aux)
        aux = np.where(aux > 255, 255, aux)

        return aux.astype(np.uint8)

    def updateSlice(self):
        if self.ctslice_rgba is None:
            return

        img = self.ctslice_rgba.copy()

        # if self.seeds is not None:
        #     if self.mode_draw:
        #         if self.contour_mode == 'fill':
        #             self.composeRgba(img, self.seeds,
        #                              self.seeds_colortable)
        #         elif self.contour_mode == 'contours':
        #             self.get_contours(img, self.seeds)
        #     else:
        #         self.overRgba(img, self.seeds,
        #                       self.seeds_colortable)

        if self.contours is not None:
            if self.contour_mode == 'fill':
                self.composeRgba(img, self.contours,
                                 CONTOURS_COLORTABLE)

            elif self.contour_mode == 'contours':
                self.get_contours(img, self.contours)

        image = QImage(img.flatten(),
                     self.slice_size[1], self.slice_size[0],
                     QImage.Format_ARGB32).scaled(self.imagesize)
        painter = QPainter(self.image)
        painter.drawImage(0, 0, image)
        painter.end()

        self.update()

    def getSliceRGBA(self, ctslice):
        if self.cw['w'] > 0:
            mul = 255.0 / float(self.cw['w'])

        else:
            mul = 0

        lb = self.cw['c'] - self.cw['w'] / 2
        aux = (ctslice.ravel(order='F') - lb) * mul
        idxs = np.where(aux < 0)[0]
        aux[idxs] = 0
        idxs = np.where(aux > 255)[0]
        aux[idxs] = 255

        return aux.astype(np.uint8)

    def updateSliceCW(self, ctslice=None):
        if ctslice is not None:
            self.ctslice_rgba = GRAY_COLORTABLE[self.getSliceRGBA(ctslice)]

        self.updateSlice()

    def setSlice(self, ctslice=None, seeds=None, contours=None):
        ctslice = np.transpose(ctslice)
        if ctslice is not None:
            if self.show_mode in (self.SHOW_IM, self.SHOW_CONTOURS):
                # tmp = self.getSliceRGBA(ctslice)
                self.ctslice_rgba = GRAY_COLORTABLE[self.getSliceRGBA(ctslice)]
            elif self.show_mode == self.SHOW_LABELS:
                self.ctslice_rgba = LABELS_COLORTABLE[ctslice.ravel(order='F')]

        if seeds is not None:
            self.seeds = seeds.ravel(order='F')
        else:
            self.seeds = None

        if contours is not None and self.show_mode == self.SHOW_CONTOURS:
            self.contours = contours.transpose(self.act_transposition).ravel(order='F')
            # self.contours = contours.transpose(self.act_transposition)
        else:
            self.contours = None

        self.updateSlice()

    def gridPosition(self, pos):
        # return (int(pos.x() / self.grid[0]),
        #         int(pos.y() / self.grid[1]))
        return (int(pos.x()), int(pos.y()))

    def resizeSlice(self, new_slice_size=None, new_grid=None, new_image_size=None):

        if new_slice_size is not None:
            self.slice_size = new_slice_size

        if new_grid is not None:
            self.grid = new_grid

        if new_image_size is not None:
            self.imagesize = new_image_size
        else:
            # self.imagesize = QSize(int(self.slice_size[0] * self.grid[0]),
            #                        int(self.slice_size[1] * self.grid[1]))
            self.imagesize = QSize(self.slice_size[0], self.slice_size[1])
        self.image = QImage(self.imagesize, QImage.Format_RGB32)
        # self.setPixmap(QPixmap.fromImage(self.image))
        # self.setPixmap(self._pixmap.scaled(QPixmap.fromImage(self.image), Qt.KeepAspectRatio)

    def resizeEvent(self, event):
        # new_height = self.height()
        # new_grid = new_height / float(self.slice_size[1])
        # mul = new_grid / self.grid[1]
        #
        # self.grid = np.array(self.grid) * mul
        self.resizeSlice()
        self.updateSlice()

    def set_width(self, new_width):
        # new_grid = new_width / float(self.slice_size[0])
        # mul = new_grid / self.grid[0]

        # self.grid = np.array(self.grid) * mul
        self.resizeSlice()
        self.updateSlice()

    def getCW(self):
        return self.cw

    def setCW(self, val, key):
        self.cw[key] = val

    def setScrollFun(self, fun):
        self.scroll_fun = fun

    def wheelEvent(self, event):
        d = event.delta()
        nd = d / abs(d)
        if self.scroll_fun is not None:
            self.scroll_fun(-nd)


    # def selectSlice(self, value, force=False):
    #     if (value < 0) or (value >= self.n_slices):
    #         return
    #
    #     # if (value != self.actual_slice) or force:
    #         # self.saveSliceSeeds()
    #         # if self.seeds_modified:
    #         #     if self.mode == 'crop':
    #         #         self.updateCropBounds()
    #         #
    #         #     elif self.mode == 'mask':
    #         #         self.updateMaskRegion()
    #
    #     if self.contours is None:
    #         contours = None
    #
    #     else:
    #         contours = self.contours_aview[...,value]
    #
    #     # slider_val = self.n_slices - value
    #     # self.slider.setValue(slider_val)
    #     # self.slider.label.setText('Slice: %d / %d' % (slider_val, self.n_slices))
    #
    #     self.setSlice(self.img_aview[...,value],
    #                             self.seeds_aview[...,value],
    #                             contours)
    #     self.actual_slice = value