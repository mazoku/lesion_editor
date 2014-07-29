__author__ = 'tomas'


import sys

from PyQt4 import QtGui, QtCore
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar

import matplotlib.pyplot as plt

import random
import numpy as np


class TumorVisualiser(QtGui.QMainWindow):

    def __init__(self, im, labels, params):
        super(TumorVisualiser, self).__init__()

        self.im = im
        self.labels = labels
        self.show_view_1 = True
        self.show_view_2 = True
        self.params = params
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
        show_im_1_action = QtGui.QAction(QtGui.QIcon('icons/Stock graph.png'), 'im_1', self)
        show_im_1_action.triggered.connect(self.show_im_1_callback)
        self.toolbar_view_1.addAction(show_im_1_action)

        # show input data on view 2
        show_im_2_action = QtGui.QAction(QtGui.QIcon('icons/Stock graph.png'), 'im_2', self)
        show_im_2_action.triggered.connect(self.show_im_2_callback)
        self.toolbar_view_2.addAction(show_im_2_action)

        # show label data on view 1
        show_labels_1_action = QtGui.QAction(QtGui.QIcon('icons/Blue tag.png'), 'labels_1', self)
        show_labels_1_action.triggered.connect(self.show_labels_1_callback)
        self.toolbar_view_1.addAction(show_labels_1_action)

        # show label data on view 2
        show_labels_2_action = QtGui.QAction(QtGui.QIcon('icons/Blue tag.png'), 'labels_2', self)
        show_labels_2_action.triggered.connect(self.show_labels_2_callback)
        self.toolbar_view_2.addAction(show_labels_2_action)

        # show label data on view 1
        show_contours_1_action = QtGui.QAction(QtGui.QIcon('icons/Brush.png'), 'contours_1', self)
        show_contours_1_action.triggered.connect(self.show_contours_1_callback)
        self.toolbar_view_1.addAction(show_contours_1_action)

        # show label data on view 2
        show_contours_2_action = QtGui.QAction(QtGui.QIcon('icons/Brush.png'), 'contours_2', self)
        show_contours_2_action.triggered.connect(self.show_contours_2_callback)
        self.toolbar_view_2.addAction(show_contours_2_action)


        # STATUS BAR -------------------
        # self.status_bar = QtGui.QStatusBar()
        self.statusBar()

        # MAIN WINDOW -------------------
        # self.setGeometry(300, 300, 300, 200)
        self.center()
        self.setWindowTitle('Tumor Visualiser')

        self.form_widget = FormWidget(self, self.params)

        self.setCentralWidget(self.form_widget)

    def view_1_callback(self):
        self.show_view_1 = not self.show_view_1
        self.statusBar().showMessage('view_1 set to %s' % self.show_view_1)
        # print 'view_1 set to', self.show_view_1
        self.form_widget.update_figures()

    def view_2_callback(self):
        self.show_view_2 = not self.show_view_2
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
        self.statusBar().showMessage('data_2 set to contours')
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

    def __init__(self, window, params):

        self.win = window  # link to the main window
        self.im = self.win.im  # input data
        self.labels = self.win.labels  # input labeling
        self.actual_slice = 0  # index of current data slice
        self.n_slices = self.im.shape[2]  # numer of slices
        self.hypo_label = params['hypo_label']
        self.hyper_label = params['hyper_label']

        super(FormWidget, self).__init__()
        self.init_UI_form()

    def init_UI_form(self):

        # QtGui.QToolTip.setFont(QtGui.QFont('SansSerif', 10))

        self.data_1 = self.im  # data to be shown in view_1
        self.data_2 = self.labels  # data to be shown in view_2
        self.data_1_str = 'im'
        self.data_2_str = 'labels'

        self.figure = plt.figure()

        self.canvas = FigureCanvas(self.figure)

        # Just some button connected to `plot` method
        # self.button = QtGui.QPushButton('Plot')
        # self.button.clicked.connect(self.plot)

        # slider
        self.slider = QtGui.QSlider(QtCore.Qt.Vertical, self)
        self.slider.setMinimum(0)
        self.slider.setMaximum(self.n_slices - 1)
        # self.slider.setMaximum(10)
        self.slider.valueChanged[int].connect(self.slider_change)
        self.slider.setSingleStep(1)  # step for arrows
        self.slider.setPageStep(1)  # step for mouse wheel

        # set the layout
        self.slice_label = QtGui.QLabel('slice #:\n%i/%i' % (self.actual_slice + 1, self.n_slices))
        slider_layout = QtGui.QVBoxLayout()
        slider_layout.addWidget(self.slider)
        slider_layout.addWidget(self.slice_label)
        slider_frame = QtGui.QFrame()
        slider_frame.setLayout(slider_layout)

        layout = QtGui.QHBoxLayout()
        layout.addWidget(self.canvas)
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
        self.win.statusBar().showMessage('actual slice changed to %i' % value)
        self.actual_slice = value
        self.update_slice_label()
        self.update_figures()

    def update_figures(self):
        if self.win.show_view_1 and self.win.show_view_2:
            plt.figure(self.figure.number)
            plt.subplot(121)
            plt.imshow(self.data_1[:, :, self.actual_slice], 'gray', interpolation='nearest')
            #TODO: otestovat contury, potreba upravit vstupni data labels
            if self.data_1_str is 'contours':
                try:
                    ctr = self.ax.contour(self.labels[:, :, self.actual_slice] == self.hypo_label, 1, colors='b', linewidths=2)
                except:
                    pass
                try:
                    ctr = self.ax.contour(self.labels[:, :, self.actual_slice] == self.hyper_label, 1, colors='r', linewidths=2)
                except:
                    pass
            plt.title('view_1: %s' % self.data_1_str)
            plt.subplot(122)
            plt.imshow(self.data_2[:, :, self.actual_slice], 'gray', interpolation='nearest')
            plt.title('view_2: %s' % self.data_2_str)
        elif self.win.show_view_1:
            plt.figure(self.figure.number)
            plt.subplot(111)
            plt.imshow(self.data_1[:, :, self.actual_slice], 'gray', interpolation='nearest')
            plt.title('view_1: %s' % self.data_1_str)
        elif self.win.show_view_2:
            plt.figure(self.figure.number)
            plt.subplot(111)
            plt.imshow(self.data_2[:, :, self.actual_slice], 'gray', interpolation='nearest')
            plt.title('view_2: %s' % self.data_2_str)
        else:
            plt.figure(self.figure.number)
            plt.clf()

        self.canvas.draw()

    def update_slice_label(self):
        self.slice_label.setText('slice #:\n%i/%i' % (self.actual_slice + 1, self.n_slices))

    def next_slice(self):
        self.actual_slice += 1
        if self.actual_slice >= self.n_slices:
            self.actual_slice = 0


    def prev_slice(self):
        self.actual_slice -= 1
        if self.actual_slice < 0:
            self.actual_slice = self.n_slices - 1

    def on_scroll(self, event):
        ''' mouse wheel is used for setting slider value'''
        if event.button == 'up':
            self.next_slice()
        if event.button == 'down':
            self.prev_slice()
        self.slider.setValue(self.actual_slice)
        # self.slider_change(self.actual_slice)


def main():
    # parameters
    params = dict()
    params['hypo_label'] = 0
    params['hyper_label'] = 2

    # preparing data
    size = 100
    n_slices = 4
    im = np.zeros((size, size, n_slices))
    step = size / n_slices
    for i in range(n_slices):
        im[i * step:(i + 1) * step, :, i] = 1

    labels = np.zeros((size, size, n_slices))
    for i in range(n_slices):
        labels[:, i * step:(i + 1) * step, i] = 1

    # runing application
    app = QtGui.QApplication(sys.argv)
    tv = TumorVisualiser(im, labels, params)
    tv.show()
    sys.exit(app.exec_())




if __name__ == '__main__':
    main()