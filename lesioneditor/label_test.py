__author__ = 'tomas'

import numpy as np
import skimage.io as skiio
import skimage.exposure as skiexp
import sys

import matplotlib.pyplot as plt

from PyQt4.QtCore import Qt, QSize, QString, SIGNAL
from PyQt4.QtGui import QImage, QMainWindow,\
    QApplication, QSlider, QPushButton,\
    QLabel, QPixmap, QPainter, qRgba,\
    QComboBox, QIcon, QStatusBar,\
    QHBoxLayout, QVBoxLayout, QFrame, QSizePolicy


# BGRA order
GRAY_COLORTABLE = np.array([[ii, ii, ii, 255] for ii in range(256)], dtype=np.uint8)

class ImageViewer(QMainWindow):
    def __init__(self, img):
        super(ImageViewer, self).__init__()

        self.image = img

        # viewer frame
        self.frame_viewer = QFrame()
        self.view_L = SliceBox(self.image)
        self.view_L.setFrameShape(QFrame.Box)
        # self.view_L.setSlice(self.image)
        self.view_L.setMinimumSize(QSize(1,1))
        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.view_L.setSizePolicy(sizePolicy)

        self.view_R = SliceBox(self.image)
        self.view_R.setFrameShape(QFrame.Box)
        # self.view_R.setSlice(self.image)
        self.view_R.setMinimumSize(QSize(1,1))
        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.view_R.setSizePolicy(sizePolicy)
        self.view_R.setVisible(False)

        self.layout_viewer = QHBoxLayout()
        self.layout_viewer.addWidget(self.view_L)
        self.layout_viewer.addWidget(self.view_R)
        self.frame_viewer.setLayout(self.layout_viewer)

        #tlacitkovy frame
        self.frame_btn = QFrame()
        self.btn = QPushButton('test')
        self.layout_btn = QHBoxLayout()
        self.layout_btn.addWidget(self.btn)
        self.frame_btn.setLayout(self.layout_btn)

        #centralni frame obsahujici frame s viewery a frame s tlacitkem
        self.frame_c = QFrame()
        self.layout_v = QVBoxLayout()
        self.layout_v.addWidget(self.frame_viewer)
        self.layout_v.addWidget(self.frame_btn)

        self.frame_c.setLayout(self.layout_v)

        self.setCentralWidget(self.frame_c)

        self.two_views = False

        self.btn.clicked.connect(self.btn_callback)

    def btn_callback(self):
        self.two_views = not self.two_views
        if self.two_views:
            self.view_R.setVisible(True)
            new_w = self.frame_viewer.width() / 2
            new_h = self.frame_viewer.height() / 2
            self.view_L.resizeSlice(new_w, new_h)
            self.view_R.resizeSlice(new_w, new_h)
        else:
            self.view_R.setVisible(False)

    def resizeEvent(self, event):
        # TODO: oba labely at maji stejnou velikost
        pass
         # if self.two_views:
         #    new_w = self.frame_viewer.width() / 2
         #    new_h = self.frame_viewer.height() / 2
         #    # new_w1 = self.width() / 2
         #    # new_h1 = self.height() / 2
         #    self.view_L.resizeSlice(new_w, new_h)
         #    self.view_R.resizeSlice(new_w, new_h)


class SliceBox(QLabel):

    def __init__(self, img):

        QLabel.__init__(self)

        self.setSlice(img)

        self.qimage = QImage(self.image.repeat(4),
                     self.imageshape[1], self.imageshape[0],
                     QImage.Format_RGB32).scaled(self.imagesize, aspectRatioMode=Qt.KeepAspectRatio)
        self.setPixmap(QPixmap.fromImage(self.qimage))
        self.orig_pixmap = self.pixmap().copy()


    def resizeSlice(self, w, h):
        self.setPixmap(self.orig_pixmap.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation))


    def resizeEvent(self, event):
        self.resizeSlice(self.width(), self.height())


    def updateSlice(self):
        if self.ctslice_rgba is None:
            return

        img = self.ctslice_rgba.copy()

        self.qimage = QImage(img.flatten(),
                      self.imageshape[1], self.imageshape[0],
                      QImage.Format_ARGB32).scaled(self.imagesize, aspectRatioMode=Qt.KeepAspectRatio)
        self.setPixmap(QPixmap.fromImage(self.qimage))


    def getSliceRGBA(self, ctslice):
        aux = ctslice.ravel(order='F')
        return aux.astype(np.uint8)


    def setSlice(self, ctslice=None, seeds=None, contours=None):

        if ctslice is not None:
            # ctslice = np.transpose(ctslice)
            self.ctslice_rgba = GRAY_COLORTABLE[self.getSliceRGBA(ctslice)]

        self.image = ctslice
        self.imageshape = ctslice.shape

        self.imagesize = QSize(self.imageshape[0], self.imageshape[1])

        self.updateSlice()


################################################################################
################################################################################
if __name__ == '__main__':
    fname = '/home/tomas/Dropbox/images/puma.png'
    img = skiio.imread(fname, as_grey=True)
    img = skiexp.rescale_intensity(img, (0,1), (0,255)).astype(np.uint8)

    app = QApplication(sys.argv)
    win = ImageViewer(img)
    win.show()
    sys.exit(app.exec_())