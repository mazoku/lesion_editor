__author__ = 'tomas'

import sys
from PyQt4 import QtGui, QtCore
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import numpy as np

# Main widget containing figures etc
class Form_widget(QtGui.QWidget):

    def __init__(self, window, cc):

        self.win = window  # link to the main window
        self.cc = cc  # link to the computational core
        # self.im = self.win.data  # input data
        # self.im = self.cc.data_1.data  # input data
        # self.labels = self.win.labels  # input labeling
        # self.labels = self.cc.data_1.labels  # input labeling
        # self.actual_slice_1 = self.win.view_1_curr_idx  # index of current data slice
        # self.actual_slice_2 = self.win.view_2_curr_idx  # index of current data slice
        # self.n_slices = self.im.shape[0]  # numer of slices
        self.healthy_label = self.win.params['healthy_label']
        self.hypo_label = self.win.params['hypo_label']
        self.hyper_label = self.win.params['hyper_label']
        self.win_w = self.win.params['win_width']
        self.win_l = self.win.params['win_level']

        super(Form_widget, self).__init__()
        self.init_UI_form()


    def init_UI_form(self):

        # QtGui.QToolTip.setFont(QtGui.QFont('SansSerif', 10))
        # self.data_l = self.im  # data to be shown in view_1
        # self.data_r = self.im  # data to be shown in view_2
        if self.win.ui.figure_1_CB.currentIndex() == 0:
            self.data_L = self.cc.data_1
        elif self.win.ui.figure_1_CB.currentIndex() == 1:
            self.data_L = self.cc.data_2
        if self.win.ui.figure_2_CB.currentIndex() == 0:
            self.data_R = self.cc.data_1
        elif self.win.ui.figure_1_CB.currentIndex() == 1:
            self.data_R = self.cc.data_2

        self.data_L_str = 'im'
        self.data_R_str = 'im'

        self.figure = plt.figure()
        self.axes = self.figure.add_axes([0, 0, 1, 1])

        self.canvas = FigureCanvas(self.figure)

        layout = QtGui.QVBoxLayout()
        layout.addWidget(self.canvas, 1)
        self.setLayout(layout)

        # conenction to wheel events
        self.canvas.mpl_connect('scroll_event', self.on_scroll)

        self.update_figures()


    def label2rgb(self, slice):
        r = slice == self.hyper_label
        g = slice == self.healthy_label
        b = slice == self.hypo_label
        slice_rgb = np.dstack((r, g, b))

        return slice_rgb


    def window_slice(self, ctslice):
        if self.win_w > 0:
            mul = 255. / float(self.win_w)
        else:
            mul = 0

        lb =self.win_l - self.win_w / 2
        # aux = (ctslice.ravel(order='F') - lb) * mul
        aux = (ctslice - lb) * mul
        # idxs = np.where(aux < 0)[0]
        # aux[idxs] = 0
        # idxs = np.where(aux > 255)[0]
        # aux[idxs] = 255
        aux = np.where(aux < 0, 0, aux)
        aux = np.where(aux > 255, 255, aux)

        return aux.astype(np.uint8)


    def update_figures(self):
        # setting the minimal and maximal to scale luminance data
        if self.data_L_str is 'labels':
            vmin1 = self.data_L.labels.min()
            vmax1 = self.data_L.labels.max()
            slice_L = self.label2rgb(self.data_L.labels[self.win.view_L_curr_idx, :, :])
        else:
            vmin1 = 0
            vmax1 = 255
            # slice_L = self.data_L.labels[self.win.view_L_curr_idx, :, :]
            slice_L = self.window_slice(self.data_L.data[self.win.view_L_curr_idx, :, :])
        if self.data_R_str is 'labels':
            vmin2 = self.data_R.labels.min()
            vmax2 = self.data_R.labels.max()
            # try:
            #     slice = self.data_R.labels[self.win.view_2_curr_idx, :, :]
            #     heal = np.where(slice == self.healthy_label)
                # ok_labels = self.data_R.labels_v[self.cc.filtered_idxs]
                # slice_R = np.where(np.in1d(slice, ok_labels).reshape(slice.shape), slice, -1)
                # slice_R = np.where(heal, self.healthy_label, slice_R)
                # slice_R = self.label2rgb(slice_R)
            # except:
            #     slice_R = self.label2rgb(self.data_2[self.actual_slice, :, :])
            slice_R = self.label2rgb(self.data_R.labels[self.win.view_R_curr_idx, :, :])
        else:
            vmin2 = 0
            vmax2 = 255
            if self.win.disp_smoothed:
                vmax2 = self.labels.max()
            slice_R = self.window_slice(self.data_R.data[self.win.view_R_curr_idx, :, :])

        # preparing slice ---------------------------------------------------
        # if both views are enabled
        if self.win.show_view_1 and self.win.show_view_2:
            plt.figure(self.figure.number)
            plt.subplot(121)
            self.figure.gca().cla()  # clearing the contours, just to be sure
            plt.imshow(slice_L, 'gray', interpolation='nearest', vmin=vmin1, vmax=vmax1)
            # displaying contours if desirable
            if self.data_L_str is 'contours':
                self.draw_contours(self.data_L.labels[self.win.view_L_curr_idx, :, :])
            plt.title('view_L: %s' % self.data_L_str)

            plt.subplot(122)
            self.figure.gca().cla()  # clearing the contours, just to be sure
            plt.imshow(slice_R, 'gray', interpolation='nearest', vmin=vmin2, vmax=vmax2)
            # plt.imshow(slice_2, interpolation='nearest')
            # displaying contours if desirable
            if self.data_R_str is 'contours':
                self.draw_contours(self.data_R.labels[self.win.view_R_curr_idx, :, :])
            plt.title('view_R: %s' % self.data_R_str)

        # if only the first view is enabled
        elif self.win.show_view_1:
            plt.figure(self.figure.number)
            plt.subplot(111)
            self.figure.gca().cla()  # clearing the contours, just to be sure
            plt.imshow(slice_L, 'gray', interpolation='nearest', vmin=vmin1, vmax=vmax1)
            if self.data_L_str is 'contours':
                self.draw_contours(self.data_L.labels[self.win.view_L_curr_idx, :, :])
            plt.title('view_L: %s' % self.data_L_str)

        # if only the second view is enabled
        elif self.win.show_view_2:
            plt.figure(self.figure.number)
            plt.subplot(111)
            self.figure.gca().cla()  # clearing the contours, just to be sure
            plt.imshow(slice_R, 'gray', interpolation='nearest', vmin=vmin2, vmax=vmax2)
            if self.data_R_str is 'contours':
                self.draw_contours(self.data_R.labels[self.win.view_R_curr_idx, :, :])
            plt.title('view_R: %s' % self.data_R_str)
        else:
            plt.figure(self.figure.number)
            plt.clf()

        self.canvas.draw()


    def draw_contours(self, labels):
        plt.hold(True)
        try:
            self.figure.gca().contour(labels == self.healthy_label, [0.1], colors='g', linewidths=2)
            plt.hold(True)
        except:
            print 'contour fail: ', sys.exc_info()[0]
        try:
            # self.figure.gca().contour(self.labels[self.actual_slice, :, :] == self.hypo_label, [0.2], colors='b', linewidths=2)
            self.figure.gca().contour(labels == self.hypo_label, [0.1], colors='b', linewidths=2)
            plt.hold(True)
        except:
            print 'contour fail: ', sys.exc_info()[0]
        try:
            # self.figure.gca().contour(self.labels[self.actual_slice, :, :] == self.hyper_label, [0.2], colors='r', linewidths=2)
            self.figure.gca().contour(labels == self.hyper_label, [0.1], colors='r', linewidths=2)
            plt.hold(False)
        except:
            print 'contour fail: ', sys.exc_info()[0]

    # def update_slice_label(self):
    #     self.slice_label.setText('slice #: %i/%i' % (self.actual_slice + 1, self.n_slices))


    def next_slice(self):
        if self.win.view_L_curr_idx < self.data_L.n_slices - 1:
            self.win.view_L_curr_idx += 1
        if self.win.view_R_curr_idx < self.data_R.n_slices - 1:
            self.win.view_R_curr_idx += 1


    def prev_slice(self):
        if self.win.view_L_curr_idx > 0:
            self.win.view_L_curr_idx -= 1
        if self.win.view_R_curr_idx > 0:
            self.win.view_R_curr_idx -= 1


    def scroll_next(self):
        self.next_slice()
        self.update_figures()


    def scroll_prev(self):
        self.prev_slice()
        self.update_figures()


    def on_scroll(self, event):
        '''mouse wheel is used for setting slider value'''
        if event.button == 'up':
            self.next_slice()
        if event.button == 'down':
            self.prev_slice()
        # self.slider.setValue(self.actual_slice)
        # self.slider_change(self.actual_slice)
        self.update_figures()
        self.win.slice_change(self.win.view_L_curr_idx)