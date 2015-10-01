__author__ = 'tomas'

import sys

from PyQt4 import QtGui, QtCore
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas

import io3d

# from simple_viewer import Ui_MainWindow
from simple_viewer import Ui_Form

# class Viewer_3D(QtGui.QMainWindow):
class Viewer_3D(QtGui.QWidget):
    """Main class of the programm."""

    def __init__(self, data, range=False, window_data=False, win_l=50, win_w=350):
        QtGui.QWidget.__init__(self)
        # self.ui = Ui_MainWindow()
        self.ui = Ui_Form()
        self.ui.setupUi(self)

        self.actual_slice = 0
        self.data = data
        self.n_slices = self.data.shape[0]
        self.range = range

        self.window_data = window_data
        self.win_l = win_l
        self.win_w = win_w

        # seting up the figure
        self.figure = plt.figure()
        self.axes = self.figure.add_axes([0, 0, 1, 1])
        self.canvas = FigureCanvas(self.figure)

        # conenction the wheel events
        self.canvas.mpl_connect('scroll_event', self.on_scroll)

        self.update_figures()

        data_viewer_layout = QtGui.QHBoxLayout()
        data_viewer_layout.addWidget(self.canvas)
        self.ui.viewer_F.setLayout(data_viewer_layout)

        # seting up the range of the scrollbar to cope with the number of slices
        self.ui.slice_scrollB.setMaximum(self.n_slices - 1)

        # connecting slider
        self.ui.slice_scrollB.valueChanged.connect(self.slider_changed)

    def update_figures(self):
        if self.range:
            vmin = self.data.min()
            vmax = self.data.max()
        elif self.window_data:
            vmin = self.win_l - self.win_w / 2
            vmax = self.win_l + self.win_w / 2
        else:
            vmin = 0
            vmax = 255
        slice = self.data[self.actual_slice, :, :]

        plt.figure(self.figure.number)
        plt.subplot(111)
        plt.imshow(slice, 'gray', interpolation='nearest', vmin=vmin, vmax=vmax)
        self.canvas.draw()

    def slider_changed(self, val):
        self.slice_change(val)
        self.actual_slice = val
        self.update_figures()

    def slice_change(self, val):
        self.ui.slice_scrollB.setValue(val)
        self.ui.slice_number_LBL.setText('slice # = %i' % (val + 1))

    def next_slice(self):
        self.actual_slice += 1
        if self.actual_slice >= self.n_slices:
            self.actual_slice = 0

    def prev_slice(self):
        self.actual_slice -= 1
        if self.actual_slice < 0:
            self.actual_slice = self.n_slices - 1

    def on_scroll(self, event):
        '''mouse wheel is used for setting slider value'''
        if event.button == 'down':
            self.next_slice()
        if event.button == 'up':
            self.prev_slice()
        self.update_figures()
        self.slice_change(self.actual_slice)

    def keyPressEvent(self, QKeyEvent):
        if QKeyEvent.key() == QtCore.Qt.Key_Escape:
            self.close()

################################################################################
################################################################################
if __name__ == '__main__':

    # fname = '/home/tomas/Data/liver_segmentation/tryba/data_other/org-exp_183_46324212_venous_5.0_B30f-.pklz'
    fname = '/home/tomas/Data/liver_segmentation/tryba/data_other/org-exp_183_46324212_arterial_5.0_B30f-.pklz'

    dr = io3d.DataReader()
    datap = dr.Get3DData(fname, dataplus_format=True)

    app = QtGui.QApplication(sys.argv)
    viewer = Viewer_3D(datap['data3d'], window_data=True)
    viewer.show()
    sys.exit(app.exec_())