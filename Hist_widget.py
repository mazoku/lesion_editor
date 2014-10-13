__author__ = 'tomas'

import sys
from PyQt4 import QtGui, QtCore
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import numpy as np

# Main widget containing figures etc
class Hist_widget(QtGui.QWidget):

    def __init__(self, window):

        self.win = window  # link to the main window
        self.im = self.win.im  # input data
        self.labels = self.win.labels  # input labeling
        self.actual_slice = 0  # index of current data slice
        self.n_slices = self.im.shape[2]  # numer of slices
        self.healthy_label = self.win.healthy_label
        self.hypo_label = self.win.hypo_label
        self.hyper_label = self.win.hyper_label

        super(Hist_widget, self).__init__()
        self.init_UI_form()

    def init_UI_form(self):
        self.data_1 = self.im  # data to be shown in view_1
        self.data_2 = self.labels  # data to be shown in view_2


        self.figure = plt.figure()
        self.axes = self.figure.add_axes([0, 0, 1, 1])

        self.canvas = FigureCanvas(self.figure)

        layout = QtGui.QVBoxLayout()
        layout.addWidget(self.canvas, 1)
        self.setLayout(layout)

        # conenction to wheel events
        self.canvas.mpl_connect('scroll_event', self.on_scroll)

        self.update_figures()


    def update_figures(self):

    #TODO: prepracovat a ozkouset histogramy

    ints = data[np.nonzero(mask)]
    hist, bins = skiexp.histogram(ints, nbins=256)
    if simple_estim:
        mu, sigma = scista.norm.fit(ints)
    else:
        ints = data[np.nonzero(mask)]

        n_pts = mask.sum()
        perc_in = n_pts * perc

        peak_idx = np.argmax(hist)
        n_in = hist[peak_idx]
        win_width = 0

        while n_in < perc_in:
            win_width += 1
            n_in = hist[peak_idx - win_width:peak_idx + win_width].sum()

        idx_start = bins[peak_idx - win_width]
        idx_end = bins[peak_idx + win_width]
        inners_m = np.logical_and(ints > idx_start, ints < idx_end)
        inners = ints[np.nonzero(inners_m)]

        # liver pdf -------------
        mu = bins[peak_idx] + params['hack_healthy_mu']
        sigma = k_std_l * np.std(inners) + params['hack_healthy_sigma']

    rv = scista.norm(mu, sigma)

    if show_me:
        plt.figure()
        plt.subplot(211)
        plt.plot(bins, hist)
        plt.title('histogram with max peak')
        plt.hold(True)
        plt.plot([mu, mu], [0, hist.max()], 'g')
        ax = plt.axis()
        plt.axis([0, 256, ax[2], ax[3]])
        # plt.subplot(212), plt.plot(bins, rv_l.pdf(bins), 'g')
        x = np.arange(0, 256, 0.1)
        plt.subplot(212), plt.plot(x, rv.pdf(x), 'g')
        plt.hold(True)
        plt.plot(mu, rv.pdf(mu), 'go')
        ax = plt.axis()
        plt.axis([0, 256, ax[2], ax[3]])
        plt.title('estimated normal pdf of healthy parenchym')

        self.canvas.draw()

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