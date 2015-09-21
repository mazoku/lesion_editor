__author__ = 'tomas'
import sys
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas

from PyQt4.QtGui import QApplication, QFrame, QVBoxLayout
from PyQt4.QtCore import Qt

import numpy as np
import skimage.exposure as skiexp

from area_hist_widget_GUI import Ui_Frame

class AreaHistWidget(QFrame):

    def __init__(self, data=None):
        # Initialize the object as a QWidget
        QFrame.__init__(self)

        self.ui = Ui_Frame()
        self.ui.setupUi(self)
        # We have to set the size of the main window
        # ourselves, since we control the entire layout
        self.setMinimumSize(400, 250)
        self.setWindowTitle('Histogram of selected area')

        self.data = data

        if data is not None:
            self.set_data(data)

        self.init_UI_form()

    def set_data(self, data):
        if data is None:
            return

        self.data = data

        self.hist, self.bins = skiexp.histogram(self.data, nbins=256)
        self.update_figures()

        mean = self.data.mean()
        std = self.data.std()
        self.ui.mean_LBL.setText('%.1f' % mean)
        self.ui.std_LBL.setText('%.1f' % std)

    def init_UI_form(self):
        self.figure = plt.figure()
        # self.axes = self.figure.add_axes([0, 0, 1, 1])
        self.axes = self.figure.add_axes()

        self.canvas = FigureCanvas(self.figure)

        layout = QVBoxLayout()
        layout.addWidget(self.canvas, 1)
        self.ui.hist_F.setLayout(layout)

        # compute histogram
        if self.data is not None:
            self.hist, self.bins = skiexp.histogram(self.data, nbins=256)
            self.update_figures()

    def update_figures(self):
        plt.figure(self.figure.number)

        plt.plot(self.bins, self.hist, 'b')
        ax = plt.axis()
        plt.axis([ax[0]-5, ax[1]+5, ax[2], ax[3]])
        plt.hold(False)

        self.canvas.draw()

    # def keyPressEvent(self, QKeyEvent):
    #     print 'hist widget key event: ',
    #     if QKeyEvent.key() == Qt.Key_Escape:
    #         print 'Escape'
    #         self.close()


################################################################################
################################################################################
if __name__ == '__main__':
# runing application -------------------------
    app = QApplication(sys.argv)
    ahw = AreaHistWidget()
    data = np.array([1, 2, 3, 4, 5, 2, 3, 4, 3])
    ahw.set_data(data)
    ahw.show()
    sys.exit(app.exec_())