__author__ = 'tomas'

import numpy as np
import skimage.io as skiio
import skimage.exposure as skiexp
import sys

from PyQt4.QtCore import Qt, QSize, QString, SIGNAL
from PyQt4.QtGui import QImage, QMainWindow,\
    QApplication, QSlider, QPushButton,\
    QLabel, QPixmap, QPainter, qRgba,\
    QComboBox, QIcon, QStatusBar,\
    QHBoxLayout, QVBoxLayout, QFrame, QSizePolicy


# BGRA order
GRAY_COLORTABLE = np.array([[ii, ii, ii, 255] for ii in range(256)],
                           dtype=np.uint8)

class ImageViewer(QMainWindow):
    def __init__(self, img):
        super(ImageViewer, self).__init__()

        self.image = img

        # viewer frame
        self.frame_viewer = QFrame()
        self.view_L = SliceBox(self.image.shape)
        self.view_L.setSlice(self.image)
        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        sizePolicy.setHeightForWidth(True)
        self.view_L.setSizePolicy(sizePolicy)
        self.view_R = SliceBox(self.image.shape)
        self.view_R.setSlice(self.image)
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
            # new_size = self.view_L.size() / 2
            self.view_R.setVisible(True)
        else:
            self.view_R.setVisible(False)
        # print self.view_L.size(), ' -> ', new_size
        # self.view_L.resizeSlice(new_image_size=new_size)


class SliceBox(QLabel):

    def __init__(self, sliceSize, mode='seeds'):

        QLabel.__init__(self)

        self.imagesize = QSize(sliceSize[0], sliceSize[1])
        self.slice_size = sliceSize
        self.ctslice_rgba = None

        self.image = QImage(self.imagesize, QImage.Format_RGB32)
        self.setPixmap(QPixmap.fromImage(self.image))
        # self.setScaledContents(True)


    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawImage(event.rect(), self.image)
        # painter.drawPixmap(event.rect(), self.pixmap())
        painter.end()


    def resizeSlice(self, scale=None, new_image_size=None):

        if new_image_size is not None:
            self.imagesize = new_image_size
        else:
            self.imagesize = QSize(self.slice_size[0], self.slice_size[1])

        if scale is None:
            scale = 1
            # self.imagesize *= scale
        # self.image = QImage(self.imagesize, QImage.Format_RGB32)
        new_size = (self.imagesize.width(), self.imagesize.height()) * scale
        self.setPixmap(self.pixmap().scaled(new_size[0], new_size[1], Qt.KeepAspectRatio))
        # else:
        #     self.setPixmap(QPixmap.fromImage(self.image))
        # self.updateSlice()


    def resizeEvent(self, event):
        self.resizeSlice()


    def updateSlice(self):
        if self.ctslice_rgba is None:
            return

        img = self.ctslice_rgba.copy()

        image = QImage(img.flatten(),
                     self.slice_size[1], self.slice_size[0],
                     QImage.Format_ARGB32).scaled(self.imagesize, aspectRatioMode=Qt.KeepAspectRatio)
        painter = QPainter(self.image)
        painter.drawImage(0, 0, image)
        painter.end()

        self.update()


    def getSliceRGBA(self, ctslice):
        aux = ctslice.ravel(order='F')
        return aux.astype(np.uint8)


    def setSlice(self, ctslice=None, seeds=None, contours=None):
        ctslice = np.transpose(ctslice)
        if ctslice is not None:
            self.ctslice_rgba = GRAY_COLORTABLE[self.getSliceRGBA(ctslice)]
        self.updateSlice()

    # def set_width(self, new_width):
    #     self.resizeSlice()
    #     self.updateSlice()


################################################################################
################################################################################
if __name__ == '__main__':
    fname = '/home/tomas/Dropbox/images/puma.png'
    img = skiio.imread(fname, as_grey=True)
    img = skiexp.rescale_intensity(img, (0,1), (0,255))

    app = QApplication(sys.argv)
    win = ImageViewer(img)
    win.show()
    sys.exit(app.exec_())