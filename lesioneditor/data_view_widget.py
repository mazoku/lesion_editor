__author__ = 'tomas'

import numpy as np

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

CONTOURS_COLORS = {
    1: [255, 0, 0],
    2: [0, 0, 255],
    3: [0, 255, 255],
    10: [255, 255, 0],
    11: [0, 255, 0],
    12: [0, 255, 255],
    13: [0, 0, 255],
}

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
    """
    Widget for marking reagions of interest in DICOM slices.
    """

    def __init__(self, sliceSize, grid, mode='seeds'):
        """
        Initialize SliceBox.

        Parameters
        ----------
        sliceSize : tuple of int
            Size of slice matrix.
        grid : tuple of float
            Pixel size:
            imageSize = (grid1 * sliceSize1, grid2 * sliceSize2)
        mode : str
            Editor mode.
        """

        QLabel.__init__(self)

        self.drawing = False
        self.modified = False
        self.seed_mark = None
        self.last_position = None
        self.imagesize = QSize(int(sliceSize[0] * grid[0]),
                               int(sliceSize[1] * grid[1]))
        self.grid = grid
        self.slice_size = sliceSize
        self.ctslice_rgba = None
        self.cw = {'c': 1.0, 'w': 1.0}

        self.seeds = None
        self.contours = None
        self.contours_old = None
        self.mask_points = None
        self.erase_region_button = None
        self.erase_fun = None
        self.erase_mode = 'inside'
        self.contour_mode = 'fill'
        self.scroll_fun = None

        # if mode == 'draw':
        #     self.seeds_colortable = CONTOURS_COLORTABLE
        #     self.box_buttons = BOX_BUTTONS_DRAW
        #     self.mode_draw = True
        # 
        # else:
        #     self.seeds_colortable = SEEDS_COLORTABLE
        #     self.box_buttons = BOX_BUTTONS_SEED
        #     self.mode_draw = False

        self.image = QImage(self.imagesize, QImage.Format_RGB32)
        self.setPixmap(QPixmap.fromImage(self.image))
        self.setScaledContents(True)


    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawImage(event.rect(), self.image)
        painter.end()


    # def drawSeedMark(self, x, y):
    #     xx = self.mask_points[0] + x
    #     yy = self.mask_points[1] + y
    #     idx = np.arange(len(xx))
    #     idx[np.where(xx < 0)] = -1
    #     idx[np.where(xx >= self.slice_size[0])] = -1
    #     idx[np.where(yy < 0)] = -1
    #     idx[np.where(yy >= self.slice_size[1])] = -1
    #     ii = idx[np.where(idx >= 0)]
    #     xx = xx[ii]
    #     yy = yy[ii]
    #
    #     self.seeds[yy * self.slice_size[0] + xx] = self.seed_mark


    # def drawLine(self, p0, p1):
    #     """
    #     Draw line to slice image and seed matrix.
    #
    #     Parameters
    #     ----------
    #     p0 : tuple of int
    #         Line star point.
    #     p1 : tuple of int
    #         Line end point.
    #     """
    #
    #     x0, y0 = p0
    #     x1, y1 = p1
    #     dx = np.abs(x1-x0)
    #     dy = np.abs(y1-y0)
    #     if x0 < x1:
    #         sx = 1
    #
    #     else:
    #         sx = -1
    #
    #     if y0 < y1:
    #         sy = 1
    #
    #     else:
    #         sy = -1
    #
    #     err = dx - dy
    #
    #     while True:
    #         self.drawSeedMark(x0,y0)
    #
    #         if x0 == x1 and y0 == y1:
    #             break
    #
    #         e2 = 2*err
    #         if e2 > -dy:
    #             err = err - dy
    #             x0 = x0 + sx
    #
    #         if e2 <  dx:
    #             err = err + dx
    #             y0 = y0 + sy


    # def drawSeeds(self, pos):
    #     if pos[0] < 0 or pos[0] >= self.slice_size[0] \
    #             or pos[1] < 0 or pos[1] >= self.slice_size[1]:
    #         return
    #
    #     self.drawLine(self.last_position, pos)
    #     self.updateSlice()
    #
    #     self.modified = True
    #     self.last_position = pos
    #
    #     self.update()


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


    def updateSlice(self):

        if self.ctslice_rgba is None:
            return

        img = self.ctslice_rgba.copy()

        if self.seeds is not None:
            if self.mode_draw:
                if self.contour_mode == 'fill':
                    self.composeRgba(img, self.seeds,
                                     self.seeds_colortable)
                elif self.contour_mode == 'contours':
                    self.get_contours(img, self.seeds)

            else:
                self.overRgba(img, self.seeds,
                              self.seeds_colortable)

        if self.contours is not None:
            if self.contour_mode == 'fill':
                self.composeRgba(img, self.contours,
                                 CONTOURS_COLORTABLE)

            elif self.contour_mode == 'contours':
                self.get_contours(img, self.contours)

        image = QImage(img.flatten(),
                     self.slice_size[0], self.slice_size[1],
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
        if ctslice is not None:
            self.ctslice_rgba = GRAY_COLORTABLE[self.getSliceRGBA(ctslice)]

        if seeds is not None:
            self.seeds = seeds.ravel(order='F')

        else:
            self.seeds = None

        if contours is not None:
            self.contours = contours.ravel(order='F')

        else:
            self.contours = None

        self.updateSlice()


    # def getSliceSeeds(self):
    #     if self.modified:
    #         self.modified = False
    #         return self.seeds.reshape(self.slice_size, order='F')
    #
    #     else:
    #         return None


    def gridPosition(self, pos):
        return (int(pos.x() / self.grid[0]),
                int(pos.y() / self.grid[1]))


    # mouse events
    # def mousePressEvent(self, event):
    #     if event.button() in self.box_buttons:
    #         self.drawing = True
    #         self.seed_mark = self.box_buttons[event.button()]
    #         self.last_position = self.gridPosition(event.pos())
    #
    #     elif event.button() == Qt.MiddleButton:
    #         self.drawing = False
    #         self.erase_region_button = True
    #
    #
    # def mouseMoveEvent(self, event):
    #     if self.drawing:
    #         self.drawSeeds(self.gridPosition(event.pos()))
    #
    #
    # def mouseReleaseEvent(self, event):
    #     if (event.button() in self.box_buttons) and self.drawing:
    #         self.drawSeeds(self.gridPosition(event.pos()))
    #         self.drawing = False
    #
    #     if event.button() == Qt.MiddleButton\
    #       and self.erase_region_button == True:
    #         self.eraseRegion(self.gridPosition(event.pos()),
    #                          self.erase_mode)
    #
    #         self.erase_region_button == False


    def resizeSlice(self, new_slice_size=None, new_grid=None):

        if new_slice_size is not None:
            self.slice_size = new_slice_size

        if new_grid is not None:
            self.grid = new_grid

        self.imagesize = QSize(int(self.slice_size[0] * self.grid[0]),
                               int(self.slice_size[1] * self.grid[1]))
        self.image = QImage(self.imagesize, QImage.Format_RGB32)
        self.setPixmap(QPixmap.fromImage(self.image))


    def resizeEvent(self, event):
        new_height = self.height()
        new_grid = new_height / float(self.slice_size[1])
        mul = new_grid / self.grid[1]

        self.grid = np.array(self.grid) * mul
        self.resizeSlice()
        self.updateSlice()


    # def leaveEvent(self, event):
    #     self.drawing = False


    # def enterEvent(self, event):
    #     self.drawing = False
    #     self.emit(SIGNAL('focus_slider'))


    # def setMaskPoints(self, mask):
    #     self.mask_points = mask


    def getCW(self):
        return self.cw


    def setCW(self, val, key):
        self.cw[key] = val


    # def eraseRegion(self, pos, mode):
    #     if self.erase_fun is not None:
    #         self.erase_fun(pos, mode)
    #         self.updateSlice()


    # def setEraseFun(self, fun):
    #     self.erase_fun = fun


    def setScrollFun(self, fun):
        self.scroll_fun = fun


    def wheelEvent(self, event):
        d = event.delta()
        nd = d / abs(d)
        if self.scroll_fun is not None:
            self.scroll_fun(-nd)