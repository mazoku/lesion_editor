__author__ = 'tomas'


import sys

from PyQt4 import QtGui, QtCore
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar

import matplotlib.pyplot as plt

import random
import numpy as np


class TumorVisualiser(QtGui.QMainWindow):

    def __init__(self, im, labels, healthy_label, hypo_label, hyper_label, disp_smoothed=False):
        super(TumorVisualiser, self).__init__()

        self.im = im
        self.labels = labels
        self.show_view_1 = True
        self.show_view_2 = True
        self.healthy_label = healthy_label
        self.hypo_label = hypo_label
        self.hyper_label = hyper_label
        self.disp_smoothed = disp_smoothed
        # self.data_1 = self.im
        # self.data_2 = self.labels

        self.init_UI_win()

    def init_UI_win(self):

        # TOOLBARS ----------------------
        # # menu
        # exitAction = QtGui.QAction(QtGui.QIcon('icons/Exit.png'), 'menu', self)
        # # exitAction.setShortcut('Ctrl+Q')
        # exitAction.triggered.connect(QtGui.qApp.quit)
        # self.toolbar_menu = self.addToolBar('menu')
        # self.toolbar_menu.addAction(exitAction)

        # view 1
        view_1_action = QtGui.QAction(QtGui.QIcon('icons/Eye.png'), 'view1', self)
        view_1_action.triggered.connect(self.view_1_callback)
        self.toolbar_view_1 = self.addToolBar('view1')
        self.toolbar_view_1.addAction(view_1_action)

        #view 2
        view_2_action = QtGui.QAction(QtGui.QIcon('icons/Eye.png'), 'view2', self)
        view_2_action.triggered.connect(self.view_2_callback)
        self.toolbar_view_2 = self.addToolBar('view2')
        self.toolbar_view_2.addAction(view_2_action)

        # show input data on view 1
        self.show_im_1_action = QtGui.QAction(QtGui.QIcon('icons/Stock graph.png'), 'im_1', self)
        self.show_im_1_action.triggered.connect(self.show_im_1_callback)
        self.toolbar_view_1.addAction(self.show_im_1_action)

        # show input data on view 2
        self.show_im_2_action = QtGui.QAction(QtGui.QIcon('icons/Stock graph.png'), 'im_2', self)
        self.show_im_2_action.triggered.connect(self.show_im_2_callback)
        self.toolbar_view_2.addAction(self.show_im_2_action)

        # show label data on view 1
        self.show_labels_1_action = QtGui.QAction(QtGui.QIcon('icons/Blue tag.png'), 'labels_1', self)
        self.show_labels_1_action.triggered.connect(self.show_labels_1_callback)
        self.toolbar_view_1.addAction(self.show_labels_1_action)

        # show label data on view 2
        self.show_labels_2_action = QtGui.QAction(QtGui.QIcon('icons/Blue tag.png'), 'labels_2', self)
        self.show_labels_2_action.triggered.connect(self.show_labels_2_callback)
        self.toolbar_view_2.addAction(self.show_labels_2_action)

        # show contours data on view 1
        self.show_contours_1_action = QtGui.QAction(QtGui.QIcon('icons/Brush.png'), 'contours_1', self)
        self.show_contours_1_action.triggered.connect(self.show_contours_1_callback)
        self.toolbar_view_1.addAction(self.show_contours_1_action)

        # show contours data on view 2
        self.show_contours_2_action = QtGui.QAction(QtGui.QIcon('icons/Brush.png'), 'contours_2', self)
        self.show_contours_2_action.triggered.connect(self.show_contours_2_callback)
        self.toolbar_view_2.addAction(self.show_contours_2_action)


        # STATUS BAR -------------------
        # self.status_bar = QtGui.QStatusBar()
        self.statusBar()

        # MAIN WINDOW -------------------
        # self.setGeometry(300, 300, 300, 200)
        self.center()
        self.setWindowTitle('Tumor Visualiser')

        self.form_widget = FormWidget(self)

        self.setCentralWidget(self.form_widget)

    def view_1_callback(self):
        self.show_view_1 = not self.show_view_1

        # enabling and disabling other toolbar icons
        self.show_im_1_action.setEnabled(not self.show_im_1_action.isEnabled())
        self.show_labels_1_action.setEnabled(not self.show_labels_1_action.isEnabled())
        self.show_contours_1_action.setEnabled(not self.show_contours_1_action.isEnabled())

        self.statusBar().showMessage('view_1 set to %s' % self.show_view_1)
        # print 'view_1 set to', self.show_view_1
        self.form_widget.update_figures()

    def view_2_callback(self):
        self.show_view_2 = not self.show_view_2

        # enabling and disabling other toolbar icons
        self.show_im_2_action.setEnabled(not self.show_im_2_action.isEnabled())
        self.show_labels_2_action.setEnabled(not self.show_labels_2_action.isEnabled())
        self.show_contours_2_action.setEnabled(not self.show_contours_2_action.isEnabled())

        self.statusBar().showMessage('view_2 set to %s' % self.show_view_2)
        # print 'view_2 set to', self.show_view_2
        self.form_widget.update_figures()

    def show_im_1_callback(self):
        # print 'data_1 set to im'
        self.statusBar().showMessage('data_1 set to im')
        self.form_widget.data_1 = self.im
        self.form_widget.data_1_str = 'im'
        self.form_widget.update_figures()

    def show_im_2_callback(self):
        # print 'data_2 set to im'
        self.statusBar().showMessage('data_2 set to im')
        if self.disp_smoothed:
            self.form_widget.data_2 = self.labels
        else:
            self.form_widget.data_2 = self.im
        self.form_widget.data_2_str = 'im'
        self.form_widget.update_figures()

    def show_labels_1_callback(self):
        # print 'data_1 set to labels'
        self.statusBar().showMessage('data_1 set to labels')
        self.form_widget.data_1 = self.labels
        self.form_widget.data_1_str = 'labels'
        self.form_widget.update_figures()

    def show_labels_2_callback(self):
        # print 'data_2 set to labels'
        self.statusBar().showMessage('data_2 set to labels')
        self.form_widget.data_2 = self.labels
        self.form_widget.data_2_str = 'labels'
        self.form_widget.update_figures()

    def show_contours_1_callback(self):
        # print 'data_2 set to contours'
        self.statusBar().showMessage('data_1 set to contours')
        self.form_widget.data_1 = self.im
        self.form_widget.data_1_str = 'contours'
        self.form_widget.update_figures()

    def show_contours_2_callback(self):
        # print 'data_2 set to contours'
        self.statusBar().showMessage('data_2 set to contours')
        self.form_widget.data_2 = self.im
        self.form_widget.data_2_str = 'contours'
        self.form_widget.update_figures()

    def center(self):
        qr = self.frameGeometry()
        cp = QtGui.QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

# Main widget containing figures etc
class FormWidget(QtGui.QWidget):

    def __init__(self, window):

        self.win = window  # link to the main window
        self.im = self.win.im  # input data
        self.labels = self.win.labels  # input labeling
        self.actual_slice = 0  # index of current data slice
        self.n_slices = self.im.shape[2]  # numer of slices
        self.healthy_label = self.win.healthy_label
        self.hypo_label = self.win.hypo_label
        self.hyper_label = self.win.hyper_label

        super(FormWidget, self).__init__()
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
        self.slider = QtGui.QSlider(QtCore.Qt.Horizontal, self)
        self.slider.setMinimum(0)
        self.slider.setMaximum(self.n_slices - 1)
        # self.slider.setMaximum(10)
        self.slider.valueChanged[int].connect(self.slider_change)
        self.slider.setSingleStep(1)  # step for arrows
        self.slider.setPageStep(1)  # step for mouse wheel

        # set the layout
        self.slice_label = QtGui.QLabel('slice #: %i/%i' % (self.actual_slice + 1, self.n_slices))
        slider_layout = QtGui.QHBoxLayout()
        # self.slice_label = QtGui.QLabel('slice #:\n%i/%i' % (self.actual_slice + 1, self.n_slices))
        # slider_layout = QtGui.QVBoxLayout()
        slider_layout.addWidget(self.slice_label)
        slider_layout.addWidget(self.slider)
        slider_frame = QtGui.QFrame()
        slider_frame.setLayout(slider_layout)

        # layout = QtGui.QHBoxLayout()
        layout = QtGui.QVBoxLayout()
        layout.addWidget(self.canvas, 1)
        layout.addWidget(slider_frame)
        # layout.addWidget(self.button)
        self.setLayout(layout)

        # # BUTTONS -----------------------
        # btn = QtGui.QPushButton('Button', self)
        # btn.setToolTip('This is a <b>QPushButton</b> widget')
        # btn.resize(btn.sizeHint())
        # btn.move(50, 50)

        # LAYOUT ------------------------
        # main_hbox = QtGui.QHBoxLayout()
        # main_hbox.addStretch(1)
        # main_hbox.addWidget(self.toolbar_menu)
        #
        # view_vbox = QtGui.QVBoxLayout()
        # view_vbox.addStretch(1)
        # view_vbox.addWidget(self.toolbar_view_1)
        # view_vbox.addWidget(self.toolbar_view_2)

        # conenction to wheel events
        self.canvas.mpl_connect('scroll_event', self.on_scroll)

        self.update_figures()

    def slider_change(self, value):
        # self.win.status_bar.showMessage('slider changed to %i' % value)
        # self.win.statusBar().showMessage('actual slice changed to %i' % value)
        self.actual_slice = value
        self.update_slice_label()
        self.update_figures()

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
            slice_1 = self.label2rgb(self.data_1[:, :, self.actual_slice])
        else:
            vmin1 = 0
            vmax1 = 255
            slice_1 = self.data_1[:, :, self.actual_slice]
        if self.data_2_str is 'labels':
            vmin2 = self.data_2.min()
            vmax2 = self.data_2.max()
            slice_2 = self.label2rgb(self.data_2[:, :, self.actual_slice])
        else:
            vmin2 = 0
            vmax2 = 255
            if self.win.disp_smoothed:
                vmax2 = self.labels.max()
            slice_2 = self.data_2[:, :, self.actual_slice]

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

        self.canvas.draw()

    def draw_contours(self):
        try:
            self.figure.gca().contour(self.labels[:, :, self.actual_slice] == self.healthy_label, [0.5], colors='g', linewidths=2)
        except:
            print 'contour fail: ', sys.exc_info()[0]
        try:
            self.figure.gca().contour(self.labels[:, :, self.actual_slice] == self.hypo_label, [0.5], colors='b', linewidths=2)
        except:
            print 'contour fail: ', sys.exc_info()[0]
        try:
            self.figure.gca().contour(self.labels[:, :, self.actual_slice] == self.hyper_label, [0.5], colors='r', linewidths=2)
        except:
            print 'contour fail: ', sys.exc_info()[0]

    def update_slice_label(self):
        self.slice_label.setText('slice #: %i/%i' % (self.actual_slice + 1, self.n_slices))

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
        if event.button == 'up':
            self.next_slice()
        if event.button == 'down':
            self.prev_slice()
        self.slider.setValue(self.actual_slice)
        # self.slider_change(self.actual_slice)


def run(im, labels, healthy_label, hypo_label, hyper_label, slice_axis=2, disp_smoothed=False):
    if slice_axis == 0:
        im = np.transpose(im, (1, 2, 0))
        labels = np.transpose(labels, (1, 2, 0))
    app = QtGui.QApplication(sys.argv)
    tv = TumorVisualiser(im, labels, healthy_label, hypo_label, hyper_label, disp_smoothed)
    tv.show()
    sys.exit(app.exec_())

def main():
    # parameters
    params = dict()
    # params['hypo_label'] = 1
    # params['hyper_label'] = 2
    healthy_label = 0
    hypo_label = 1
    hyper_label = 2

    # preparing data
    size = 100
    n_slices = 4
    im = np.zeros((size, size, n_slices))
    step = size / n_slices
    for i in range(n_slices):
        im[i * step:(i + 1) * step, :, i] = 150

    labels = np.zeros((size, size, n_slices))
    for i in range(n_slices):
        if np.mod(i, 2) == 0:
            lab = 1
        else:
            lab = 2
        labels[:, i * step:(i + 1) * step, i] = lab

    # runing application
    run(im, labels, healthy_label, hypo_label, hyper_label)

if __name__ == '__main__':
    main()