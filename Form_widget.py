__author__ = 'tomas'

import sys
from PyQt4 import QtGui, QtCore
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import numpy as np

# Main widget containing figures etc
class Form_widget(QtGui.QWidget):

    def __init__(self, window):

        self.win = window  # link to the main window
        self.im = self.win.data  # input data
        self.labels = self.win.labels  # input labeling
        self.actual_slice = 0  # index of current data slice
        self.n_slices = self.im.shape[0]  # numer of slices
        self.healthy_label = self.win.healthy_label
        self.hypo_label = self.win.hypo_label
        self.hyper_label = self.win.hyper_label

        super(Form_widget, self).__init__()
        self.init_UI_form()

    def init_UI_form(self):

        # QtGui.QToolTip.setFont(QtGui.QFont('SansSerif', 10))

        self.data_1 = self.im  # data to be shown in view_1
        self.data_2 = self.labels  # data to be shown in view_2
        self.data_1_str = 'im'
        self.data_2_str = 'labels'

        self.figure = plt.figure()
        self.axes = self.figure.add_axes([0, 0, 1, 1])

        self.canvas = FigureCanvas(self.figure)

        # Just some button connected to `plot` method
        # self.button = QtGui.QPushButton('Plot')
        # self.button.clicked.connect(self.plot)

        # slider
        # self.slider = QtGui.QSlider(QtCore.Qt.Vertical, self)
        # self.slider = QtGui.QSlider(QtCore.Qt.Horizontal, self)
        # self.slider.setMinimum(0)
        # self.slider.setMaximum(self.n_slices - 1)
        # # self.slider.setMaximum(10)
        # self.slider.valueChanged[int].connect(self.slider_change)
        # self.slider.setSingleStep(1)  # step for arrows
        # self.slider.setPageStep(1)  # step for mouse wheel
        #
        # # set the layout
        # self.slice_label = QtGui.QLabel('slice #: %i/%i' % (self.actual_slice + 1, self.n_slices))
        # slider_layout = QtGui.QHBoxLayout()
        # # self.slice_label = QtGui.QLabel('slice #:\n%i/%i' % (self.actual_slice + 1, self.n_slices))
        # # slider_layout = QtGui.QVBoxLayout()
        # slider_layout.addWidget(self.slice_label)
        # slider_layout.addWidget(self.slider)
        # slider_frame = QtGui.QFrame()
        # slider_frame.setLayout(slider_layout)

        # layout = QtGui.QHBoxLayout()
        layout = QtGui.QVBoxLayout()
        layout.addWidget(self.canvas, 1)
        # layout.addWidget(slider_frame)
        # layout.addWidget(self.button)
        self.setLayout(layout)

        # conenction to wheel events
        self.canvas.mpl_connect('scroll_event', self.on_scroll)

        self.update_figures()

    # def slider_change(self, value):
    #     # self.win.status_bar.showMessage('slider changed to %i' % value)
    #     # self.win.statusBar().showMessage('actual slice changed to %i' % value)
    #     self.actual_slice = value
    #     self.update_slice_label()
    #     self.update_figures()

    def label2rgb(self, slice):
        r = slice == self.hyper_label
        g = slice == self.healthy_label
        b = slice == self.hypo_label
        slice_rgb = np.dstack((r, g, b))

        return slice_rgb

    def update_figures(self):
        # setting the minimal and maximal to scale luminance data
        if self.data_1_str is 'labels':
            vmin1 = self.data_1.min()
            vmax1 = self.data_1.max()
            slice_1 = self.label2rgb(self.data_1[self.actual_slice, :, :])
        else:
            vmin1 = 0
            vmax1 = 255
            slice_1 = self.data_1[self.actual_slice, :, :]
        if self.data_2_str is 'labels':
            vmin2 = self.data_2.min()
            vmax2 = self.data_2.max()
            slice_2 = self.label2rgb(self.data_2[self.actual_slice, :, :])
        else:
            vmin2 = 0
            vmax2 = 255
            if self.win.disp_smoothed:
                vmax2 = self.labels.max()
            slice_2 = self.data_2[self.actual_slice, :, :]

        # if both views are enabled
        if self.win.show_view_1 and self.win.show_view_2:
            plt.figure(self.figure.number)
            plt.subplot(121)
            self.figure.gca().cla()  # clearing the contours, just to be sure
            plt.imshow(slice_1, 'gray', interpolation='nearest', vmin=vmin1, vmax=vmax1)
            # displaying contours if desirable
            if self.data_1_str is 'contours':
                self.draw_contours()
            plt.title('view_1: %s' % self.data_1_str)

            plt.subplot(122)
            self.figure.gca().cla()  # clearing the contours, just to be sure
            plt.imshow(slice_2, 'gray', interpolation='nearest', vmin=vmin2, vmax=vmax2)
            # displaying contours if desirable
            if self.data_2_str is 'contours':
                self.draw_contours()
            plt.title('view_2: %s' % self.data_2_str)

        # if only the first view is enabled
        elif self.win.show_view_1:
            plt.figure(self.figure.number)
            plt.subplot(111)
            self.figure.gca().cla()  # clearing the contours, just to be sure
            plt.imshow(slice_1, 'gray', interpolation='nearest', vmin=vmin1, vmax=vmax1)
            if self.data_1_str is 'contours':
                self.draw_contours()
            plt.title('view_1: %s' % self.data_1_str)

        # if only the second view is enabled
        elif self.win.show_view_2:
            plt.figure(self.figure.number)
            plt.subplot(111)
            self.figure.gca().cla()  # clearing the contours, just to be sure
            plt.imshow(slice_2, 'gray', interpolation='nearest', vmin=vmin2, vmax=vmax2)
            if self.data_2_str is 'contours':
                self.draw_contours()
            plt.title('view_2: %s' % self.data_2_str)
        else:
            plt.figure(self.figure.number)
            plt.clf()

        # creating overaly from itensity model
        if self.win.ui.prob_heal_1_CB.isChecked():
            print 'checked'

        self.canvas.draw()

    def draw_contours(self):
        try:
            self.figure.gca().contour(self.labels[self.actual_slice, :, :] == self.healthy_label, [0.5], colors='g', linewidths=2)
        except:
            print 'contour fail: ', sys.exc_info()[0]
        try:
            self.figure.gca().contour(self.labels[self.actual_slice, :, :] == self.hypo_label, [0.5], colors='b', linewidths=2)
        except:
            print 'contour fail: ', sys.exc_info()[0]
        try:
            self.figure.gca().contour(self.labels[self.actual_slice, :, :] == self.hyper_label, [0.5], colors='r', linewidths=2)
        except:
            print 'contour fail: ', sys.exc_info()[0]

    # def update_slice_label(self):
    #     self.slice_label.setText('slice #: %i/%i' % (self.actual_slice + 1, self.n_slices))

    def next_slice(self):
        self.actual_slice += 1
        if self.actual_slice >= self.n_slices:
            self.actual_slice = 0

    def prev_slice(self):
        print 'actual: ', self.actual_slice
        self.actual_slice -= 1
        print 'new: ', self.actual_slice
        if self.actual_slice < 0:
            self.actual_slice = self.n_slices - 1
            print 'remaped to: ', self.actual_slice

    def on_scroll(self, event):
        '''mouse wheel is used for setting slider value'''
        if event.button == 'up':
            self.next_slice()
        if event.button == 'down':
            self.prev_slice()
        # self.slider.setValue(self.actual_slice)
        # self.slider_change(self.actual_slice)
        self.update_figures()
        self.win.slice_change(self.actual_slice)