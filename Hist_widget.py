__author__ = 'tomas'

import sys
from PyQt4 import QtGui, QtCore
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import numpy as np
import skimage.exposure as skiexp
import scipy.stats as scista

# Main widget containing figures etc
class Hist_widget(QtGui.QWidget):

    def __init__(self, window):

        self.win = window  # link to the main window
        self.data = self.win.data  # input data
        self.mask = self.win.mask
        # self.labels = self.win.labels  # input labeling
        # self.actual_slice = 0  # index of current data slice
        # self.n_slices = self.im.shape[2]  # numer of slices
        # self.healthy_label = self.win.healthy_label
        # self.hypo_label = self.win.hypo_label
        # self.hyper_label = self.win.hyper_label
        self.rv_healthy = None
        self.rv_hypo = None
        self.rv_hyper = None

        super(Hist_widget, self).__init__()
        self.init_UI_form()


    def init_UI_form(self):
        self.figure = plt.figure()
        self.axes = self.figure.add_axes([0, 0, 1, 1])

        self.canvas = FigureCanvas(self.figure)

        layout = QtGui.QVBoxLayout()
        layout.addWidget(self.canvas, 1)
        self.setLayout(layout)

        # compute histogram
        ints = self.data[np.nonzero(self.mask)]
        self.hist, self.bins = skiexp.histogram(ints, nbins=256)

        # conenction to wheel events
        # self.canvas.mpl_connect('scroll_event', self.on_scroll)

        self.update_figures()


    def update_heal_rv(self, new_rv):
        self.rv_healthy = new_rv


    def update_hypo_rv(self, new_rv):
        self.rv_hypo = new_rv


    def update_hyper_rv(self, new_rv):
        self.rv_hyper = new_rv


    def update_figures(self):
        plt.figure(self.figure.number)
        x = np.arange(0, 256, 0.1)  # artificial x-axis
        # self.figure.gca().cla()  # clearing the figure, just to be sure

        plt.subplot(411)
        plt.plot(self.bins, self.hist, 'k')
        plt.hold(True)
        if self.rv_healthy and self.rv_hypo and self.rv_hyper:
            healthy_y = self.rv_healthy.pdf(x)
            hypo_y = self.rv_hypo.pdf(x)
            hyper_y = self.rv_hyper.pdf(x)
            y_max = max(healthy_y.max(), hypo_y.max(), hyper_y.max())
            fac = self.hist.max() / y_max

            plt.plot(x, fac * healthy_y, 'g', linewidth=2)
            plt.plot(x, fac * hypo_y, 'b', linewidth=2)
            plt.plot(x, fac * hyper_y, 'r', linewidth=2)
            plt.title('all PDFs')
        ax = plt.axis()
        plt.axis([0, 256, ax[2], ax[3]])
        plt.hold(False)

        plt.subplot(412)
        plt.plot(self.bins, self.hist, 'k')
        plt.hold(True)
        if self.rv_healthy:
            plt.plot(x, fac * self.rv_healthy.pdf(x), 'g', linewidth=2)
        plt.title('healthy PDF')
        ax = plt.axis()
        plt.axis([0, 256, ax[2], ax[3]])
        plt.hold(False)

        plt.subplot(413)
        plt.plot(self.bins, self.hist, 'k')
        plt.hold(True)
        if self.rv_hypo:
            plt.plot(x, fac * self.rv_hypo.pdf(x), 'b', linewidth=2)
        plt.title('hypodense PDF')
        ax = plt.axis()
        plt.axis([0, 256, ax[2], ax[3]])
        plt.hold(False)

        plt.subplot(414)
        plt.plot(self.bins, self.hist, 'k')
        plt.hold(True)
        if self.rv_hyper:
            plt.plot(x, fac * self.rv_hyper.pdf(x), 'r', linewidth=2)
        plt.title('hyperdense PDF')
        ax = plt.axis()
        plt.axis([0, 256, ax[2], ax[3]])
        plt.hold(False)

        self.canvas.draw()

    # def on_scroll(self, event):
    #     '''mouse wheel is used for setting slider value'''
    #     if event.button == 'up':
    #         self.next_slice()
    #     if event.button == 'down':
    #         self.prev_slice()
    #     # self.slider.setValue(self.actual_slice)
    #     # self.slider_change(self.actual_slice)
    #     self.update_figures()
    #     self.win.slice_change(self.actual_slice)