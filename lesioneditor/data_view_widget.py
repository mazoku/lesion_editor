__author__ = 'tomas'

import numpy as np
import matplotlib.pyplot as plt

import sys

from PyQt4.QtCore import Qt, QSize, QString, SIGNAL, QPoint, pyqtSignal
from PyQt4.QtGui import QImage, QDialog,\
    QApplication, QSlider, QPushButton,\
    QLabel, QPixmap, QPainter, qRgba,\
    QComboBox, QIcon, QStatusBar,\
    QHBoxLayout, QVBoxLayout, QFrame, QSizePolicy,\
    QPen, QMainWindow, QWidget

import skimage.morphology as skimor
import skimage.io as skiio

import area_hist_widget as ahw


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
    2: [0, 255, 0],
    3: [0, 0, 255],
    # 10: [255, 255, 0],
    # 11: [0, 255, 0],
    # 12: [0, 255, 255],
    # 13: [0, 0, 255],
}

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
    CONTOURLINES_COLORTABLE[key,0,3] = 0  # alpha channel of color inside a contour, 0 = no color
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

    mouseClickSignal = pyqtSignal(list, int)

    def __init__(self, main):#, sliceSize, voxel_size, main, mode='seeds'):
    # def __init__(self, sliceSize, grid, mode='seeds'):
    # def __init__(self, sliceSize, mode='seeds'):

        QLabel.__init__(self)
        self.main = main

    def setup_widget(self, sliceSize, voxel_size):
        height = 600
        vscale = voxel_size / float(np.min(voxel_size))
        grid = height / float(sliceSize[1] * vscale[1])
        grid = (grid * vscale[0], grid * vscale[1])

        # self.drawing = False
        # self.modified = False
        # self.last_position = None
        self.imagesize = QSize(int(sliceSize[0] * grid[0]),
                               int(sliceSize[1] * grid[1]))

        # self.setMouseTracking(True)
        self.mouse_cursor = [0, 0]
        self.mouse_glob = [0, 0]
        self.circle_r = 1
        self.circle_strel = np.nonzero(skimor.disk(self.circle_r))
        self.circle_m = self.circle_strel

        self.SHOW_IM = 0  # flag for showing density data
        self.SHOW_LABELS = 1  # flag for showing labels
        self.SHOW_CONTOURS = 2  # flag for showing contours

        # self.imagesize = QSize(sliceSize[0], sliceSize[1])
        self.grid = grid
        self.grid_res = np.copy(grid)
        self.slice_size = sliceSize
        self.ctslice_rgba = None
        self.cw = {'c': 1.0, 'w': 1.0}

        self.seeds = None
        self.contours = None
        self.contours_old = None
        self.mask_points = None
        # self.contour_mode = 'fill'
        # self.contour_mode = 'contours'
        self.contours_mode_is_fill = True

        self.show_mode = self.SHOW_IM

        # self.actual_slice = 0
        # self.n_slices = 0
        self.scroll_fun = None

        self.image = QImage(self.imagesize, QImage.Format_RGB32)
        self.setPixmap(QPixmap.fromImage(self.image))
        self.setScaledContents(True)

        self.actual_view = 'axial'
        # self.act_transposition = VIEW_TABLE[self.actual_view]

        self.circle_active = False
        self.ruler_active = False

        self.area_hist_widget = None

        self.setScaledContents(True)

        # self.area_hist_widget = ahw.AreaHistWidget()
        # self.area_hist_widget.setEnabled(False)
        # self.area_hist_widget.show()

    def reinit(self, sliceSize):
        self.imagesize = QSize(int(sliceSize[0] * self.grid[0]),
                               int(sliceSize[1] * self.grid[1]))
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
        self.setScaledContents(True)

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
            # if self.contour_mode == 'fill':
            if self.contours_mode_is_fill:
                self.composeRgba(img, self.contours, CONTOURS_COLORTABLE)
            # elif self.contour_mode == 'contours':
            else:
                self.get_contours(img, self.contours)

        # masking out pixels under circle
        # for i in self.circle_m:
        #     img[i, :] = [0, 0, 255, 255]

        image = QImage(img.flatten(),
                     self.slice_size[0], self.slice_size[1],
                     QImage.Format_ARGB32).scaled(self.imagesize)
        painter = QPainter(self.image)
        painter.drawImage(0, 0, image)

        if self.show_mode == self.SHOW_CONTOURS:# and self.centers is not None:
            if self.centers is not None:
                pts = self.centers.nonzero()
                pen = QPen(Qt.red, 3)
                painter.setPen(pen)
                for i in range(len(pts[0])):
                    painter.drawPoint(pts[1][i] * self.grid[0], pts[0][i] * self.grid[1])

        if self.circle_active:
            pen = QPen(Qt.red, 3)
            painter.setPen(pen)
            center_offset = 0 #0.5
            radius_offset = 0 #0.5
            painter.drawEllipse(QPoint((self.mouse_cursor[0] + center_offset) * self.grid[0], (self.mouse_cursor[1] + center_offset) * self.grid[1]),
                                (self.circle_r + radius_offset) * self.grid[0], (self.circle_r + radius_offset) * self.grid[1])
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

    def setSlice(self, ctslice=None, seeds=None, contours=None, centers=None):
        # ctslice = 200 * np.triu(np.ones(ctslice.shape, dtype=int))
        self.ctslice = ctslice
        self.centers = centers
        # ctslice = np.transpose(ctslice)
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
            self.contours = contours.transpose().ravel(order='F')#self.act_transposition).ravel(order='F')
            # self.contours = contours.transpose(self.act_transposition)
        else:
            self.contours = None

        self.updateSlice()

    def gridPosition(self, pos):
        # return (int(pos.x() / self.grid[0]),
        #         int(pos.y() / self.grid[1]))
        return (int(pos.x() / self.grid_res[0]),
                int(pos.y() / self.grid_res[1]))
        # return (int(pos.x()), int(pos.y()))

    def resizeSlice(self, new_slice_size=None, new_grid=None, new_image_size=None):

        if new_slice_size is not None:
            self.slice_size = new_slice_size

        if new_grid is not None:
            self.grid = new_grid

        if new_image_size is not None:
            self.imagesize = new_image_size
        else:
            self.imagesize = QSize(int(self.slice_size[0] * self.grid[0]),
                                   int(self.slice_size[1] * self.grid[1]))
            # self.imagesize = QSize(self.slice_size[0], self.slice_size[1])
        # self.image = QImage(self.imagesize, QImage.Format_RGB32)
        # pixmap = QPixmap.fromImage(self.image).scaled(self.size(), Qt.KeepAspectRatio)
        # self.setPixmap(pixmap)
        self.setPixmap(QPixmap.fromImage(self.image))

    def resizeEvent(self, event):
        new_height = self.height()
        new_grid_height = new_height / float(self.slice_size[1])
        mul_height = new_grid_height / self.grid[1]
        # mul_width = mul_height

        new_width = self.width()
        new_grid_width = new_width / float(self.slice_size[0])
        mul_width = new_grid_width / self.grid[0]

        # self.grid = np.array(self.grid) * mul_height
        # self.grid =  np.array(self.grid)
        # self.grid[0] *= mul_width
        # self.grid[1] *= mul_height
        self.grid_res[0] = self.grid[0] * mul_width
        self.grid_res[1] = self.grid[1] * mul_height
        # print self.grid, self.grid_res

        self.resizeSlice()
        self.updateSlice()

    def set_width(self, new_width):
        new_grid = new_width / float(self.slice_size[0])
        mul = new_grid / self.grid[0]

        self.grid = np.array(self.grid) * mul
        self.resizeSlice()
        self.updateSlice()

    def getCW(self):
        return self.cw

    def setCW(self, val, key):
        self.cw[key] = val

    def setScrollFun(self, fun):
        self.scroll_fun = fun

    def wheelEvent(self, event):
        step = 4
        d = event.delta()
        nd = d / abs(d)
        self.circle_r += step * nd
        self.circle_r = max(self.circle_r, 1)
        self.circle_strel = np.nonzero(skimor.disk(self.circle_r))
        self.updateSlice()
        # if self.scroll_fun is not None:
        #     self.scroll_fun(-nd)

    def mouseMoveEvent(self, QMouseEvent):
        center = list(self.gridPosition(QMouseEvent.pos()))
        self.mouse_cursor = center

        circle_x = (self.circle_strel[1] + center[0] - self.circle_r).astype(np.int)
        circle_y = (self.circle_strel[0] + center[1] - self.circle_r).astype(np.int)

        idx_x = (circle_x >= 0) * (circle_x < self.slice_size[0])
        idx_y = (circle_y >= 0) * (circle_y < self.slice_size[1])
        idx = idx_x * idx_y
        circle_x = circle_x[np.nonzero(idx)]
        circle_y = circle_y[np.nonzero(idx)]

        self.circle_m = np.ravel_multi_index((circle_y, circle_x), self.slice_size[::-1])

        if self.area_hist_widget is not None:
            self.circle_area_data = self.ctslice[(circle_x, circle_y)]
            self.area_hist_widget.set_data(self.circle_area_data)

        self.updateSlice()

    # def mousePressEvent(self, QMouseEvent):
    #     print 'click'
    #     coords = list(self.gridPosition(QMouseEvent.pos()))
    #     print 'pos = ', coords, ', data = ', self.ctslice[coords[0], coords[1]]
        # if self.show_mode == self.SHOW_LABELS:
        #     label = self.main.actual_data
        # return coords

    def myMousePressEvent(self, QMouseEvent):
        print 'myMousePressEvent: ',
        coords = list(self.gridPosition(QMouseEvent.pos()))
        density = self.ctslice[coords[0], coords[1]]
        print 'pos = ', coords, ', data = ', density
        self.mouseClickSignal.emit(coords, density)
        # self.mousePressEvent = None
        # self.mousePressEvent = self.myEmptyMousePressEvent

    def myEmptyMousePressEvent(self, QMouseEvevnt):
        print 'dummy mouse press event'

if __name__ == '__main__':
    from lession_editor_GUI_slim2 import Ui_MainWindow

    class Window(QMainWindow):
        def __init__(self, img, parent=None):
            QWidget.__init__(self, parent)

            v_size = np.array([1, 1, 1])

            viewer = SliceBox(img.shape, v_size)
            viewer.setCW(0, 'c')
            viewer.setCW(1, 'w')
            viewer.setSlice(img)
            viewer.circle_active = True
            viewer.setMouseTracking(True)

            layout = QHBoxLayout()
            layout.addWidget(viewer)

            ui = Ui_MainWindow()
            ui.setupUi(self)
            ui.viewer_F.setLayout(layout)

    app = QApplication(sys.argv)

    img = skiio.imread('/home/tomas/Dropbox/images/puma.png', as_grey=True)
    img = img.transpose()
    win = Window(img)
    win.show()

    sys.exit(app.exec_())