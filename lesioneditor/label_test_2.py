__author__ = 'tomas'

import sys

import skimage.io as skiio
import skimage.exposure as skiexp

import numpy as np

from PyQt4.QtGui import QApplication, QLabel, QPixmap, QImage
from PyQt4.QtCore import Qt, QSize


class SliceBox(QLabel):

    def __init__(self, img, mode='seeds'):

        QLabel.__init__(self)

        self.image = img
        self.imageshape = img.shape

        self.imagesize = QSize(self.imageshape[0], self.imageshape[1])

        self.qimage = QImage(self.image.repeat(4),
                     self.imageshape[1], self.imageshape[0],
                     QImage.Format_RGB32).scaled(self.imagesize, aspectRatioMode=Qt.KeepAspectRatio)
        self.setPixmap(QPixmap.fromImage(self.qimage))
        self.orig_pixmap = self.pixmap().copy()


    def resizeEvent(self, event):
        self.resizeSlice(self.width(), self.height())


    def resizeSlice(self, w, h):
        self.setPixmap(self.orig_pixmap.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation))


#-----------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------
fname = '/home/tomas/Dropbox/images/puma.png'
image = skiio.imread(fname, as_grey=True)
image = skiexp.rescale_intensity(image, (0,1), (0,255)).astype(np.uint8)

app = QApplication(sys.argv)
label = SliceBox(image)
label.show()

sys.exit(app.exec_())