__author__ = 'tomas'

import sys

import skimage.io as skiio
import skimage.exposure as skiexp

import numpy as np

from PyQt4.QtGui import QApplication, QLabel, QPixmap, QImage, QPainter
from PyQt4.QtCore import Qt, QSize, QRectF


class SliceBox(QLabel):

    def __init__(self, img, mode='seeds'):

        QLabel.__init__(self)

        self.image = img
        self.imageshape = img.shape
        self.aspect_ratio = self.imageshape[0] / float(self.imageshape[1])

        self.imagesize = QSize(self.imageshape[0], self.imageshape[1])

        self.qimage = QImage(self.image.repeat(4),
                     self.imageshape[1], self.imageshape[0],
                     QImage.Format_RGB32).scaled(self.imagesize, aspectRatioMode=Qt.KeepAspectRatio)
        self.setPixmap(QPixmap.fromImage(self.qimage))
        self.orig_pixmap = self.pixmap().copy()


    # def paintEvent(self, event):
    #     w = self.width()
    #     h = self.height()
    #
    #     x = (self.rect().width() - w) / 2
    #     y = (self.rect().height() - h) / 2
    #     x = max(x, 0)
    #     y = max(y, 0)
    #     target = QRectF(x, y, w, h)
    #     painter = QPainter(self)
    #     painter.drawImage(target, self.qimage)
    #     # painter.drawPixmap()
    #     painter.end()

    def resizeEvent(self, event):
        self.resizeSlice(self.width(), self.height())
        # self.image = QImage(self.cv_img.data, width, height, bytesPerLine, QImage.Format_RGB888)

    def resizeSlice(self, w, h):
        self.setPixmap(self.orig_pixmap.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        # pass
    #     if new_image_size is not None:
    #         self.imagesize = new_image_size
    #     else:
    #         self.imagesize = QSize(self.slice_size[0], self.slice_size[1])
    #
    #     if scale is None:
    #         scale = 1
    #         # self.imagesize *= scale
    #     # self.image = QImage(self.imagesize, QImage.Format_RGB32)
    #     new_size = (self.imagesize.width(), self.imagesize.height()) * scale
    #     self.setPixmap(self.pixmap().scaled(new_size[0], new_size[1], Qt.KeepAspectRatio))



#-----------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------
fname = '/home/tomas/Dropbox/images/puma.png'
image = skiio.imread(fname, as_grey=True)
image = skiexp.rescale_intensity(image, (0,1), (0,255)).astype(np.uint8)
width, height = image.shape

# qimage = QImage((width, height), QImage.Format_RGB32)

app = QApplication(sys.argv)
label = SliceBox(image)

# image = image.scaled(width, height, Qt.KeepAspectRatio)
# label.setPixmap(QPixmap.fromImage(image))

# pixmap = QPixmap()
# label.setPixmap(pixmap)
label.show()

sys.exit(app.exec_())