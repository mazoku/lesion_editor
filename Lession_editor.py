__author__ = 'tomas'

import sys
from PyQt4 import QtGui, QtCore

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage.morphology as scindimor
import scipy.ndimage.measurements as scindimea
import scipy.ndimage.interpolation as scindiint
import scipy.ndimage as scindi
import scipy.stats as scista

import skimage.segmentation as skiseg
import skimage.morphology as skimor
import skimage.filter as skifil
import skimage.exposure as skiexp
import skimage.measure as skimea
import skimage.transform as skitra

import cv2
import pygco

import tools
import py3DSeedEditor
from mayavi import mlab

import TumorVisualiser

from sklearn import metrics
from sklearn.cluster import KMeans

import pickle

import logging
logger = logging.getLogger(__name__)
logging.basicConfig()


from lession_editor_GUI import Ui_MainWindow
import Form_widget

class Lession_editor(QtGui.QMainWindow):
    """Main class of the programm."""

    def __init__(self, im, labels, healthy_label, hypo_label, hyper_label, disp_smoothed=False, parent=None):
        QtGui.QWidget.__init__(self, parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # uprava stylu pro lepsi vizualizaci splitteru
        QtGui.QApplication.setStyle(QtGui.QStyleFactory.create('Cleanlooks'))

        self.im = im
        self.labels = labels
        self.show_view_1 = True
        self.show_view_2 = True
        self.healthy_label = healthy_label
        self.hypo_label = hypo_label
        self.hyper_label = hyper_label
        self.disp_smoothed = disp_smoothed
        self.n_slices = im.shape[2]

        # nastaveni rozsahu scrollBaru podle poctu rezu
        self.ui.slice_scrollB.setMaximum(self.n_slices - 1)

        self.form_widget = Form_widget.Form_widget(self)
        data_viewer_layout = QtGui.QHBoxLayout()
        data_viewer_layout.addWidget(self.form_widget)
        self.ui.viewer_F.setLayout(data_viewer_layout)

        # connecting callbacks ----------------------------------
        self.ui.view_1_BTN.clicked.connect(self.view_1_callback)
        self.ui.view_2_BTN.clicked.connect(self.view_2_callback)

        # show image data
        self.ui.show_im_1_BTN.clicked.connect(self.show_im_1_callback)
        self.ui.show_im_2_BTN.clicked.connect(self.show_im_2_callback)

        # show label data
        self.ui.show_labels_1_BTN.clicked.connect(self.show_labels_1_callback)
        self.ui.show_labels_2_BTN.clicked.connect(self.show_labels_2_callback)

        # show contours data
        self.ui.show_contours_1_BTN.clicked.connect(self.show_contours_1_callback)
        self.ui.show_contours_2_BTN.clicked.connect(self.show_contours_2_callback)

        # connecting slider
        self.ui.slice_scrollB.valueChanged.connect(self.slider_changed)

    def wheelEvent(self, event):
        print event.delta() / 120

    def slider_changed(self, val):
        self.slice_change(val)
        self.form_widget.actual_slice = val
        self.form_widget.update_figures()

    def slice_change(self, val):
        self.ui.slice_scrollB.setValue(val)
        self.ui.slice_number_LBL.setText('slice # = %i' % (val + 1))

    def view_1_callback(self):
        self.show_view_1 = not self.show_view_1

        # enabling and disabling other toolbar icons
        self.ui.show_im_1_BTN.setEnabled(not self.ui.show_im_1_BTN.isEnabled())
        self.ui.show_labels_1_BTN.setEnabled(not self.ui.show_labels_1_BTN.isEnabled())
        self.ui.show_contours_1_BTN.setEnabled(not self.ui.show_contours_1_BTN.isEnabled())

        self.statusBar().showMessage('view_1 set to %s' % self.show_view_1)
        # print 'view_1 set to', self.show_view_1

        print 'upravit update figur'
        self.form_widget.update_figures()

    def view_2_callback(self):
        self.show_view_2 = not self.show_view_2

        # enabling and disabling other toolbar icons
        self.ui.show_im_2_BTN.setEnabled(not self.ui.show_im_2_BTN.isEnabled())
        self.ui.show_labels_2_BTN.setEnabled(not self.ui.show_labels_2_BTN.isEnabled())
        self.ui.show_contours_2_BTN.setEnabled(not self.ui.show_contours_2_BTN.isEnabled())

        self.statusBar().showMessage('view_2 set to %s' % self.show_view_2)
        # print 'view_2 set to', self.show_view_2

        print 'upravit update figur'
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


def run(im, labels, healthy_label, hypo_label, hyper_label, slice_axis=2, disp_smoothed=False):
    if slice_axis == 0:
        im = np.transpose(im, (1, 2, 0))
        labels = np.transpose(labels, (1, 2, 0))
    app = QtGui.QApplication(sys.argv)
    le = Lession_editor(im, labels, healthy_label, hypo_label, hyper_label, disp_smoothed)
    le.show()
    sys.exit(app.exec_())


################################################################################
################################################################################
if __name__ == '__main__':
    fname = ''

    # 2 hypo, 1 on the border --------------------
    # arterial 0.6mm - bad
    # fname = '/home/tomas/Data/liver_segmentation_06mm/tryba/data_other/org-exp_183_46324212_arterial_0.6_B30f-.pklz'
    # venous 0.6mm - good
    # fname = '/home/tomas/Data/liver_segmentation_06mm/tryba/data_other/org-exp_183_46324212_venous_0.6_B20f-.pklz'
    # venous 5mm - ok, but wrong approach
    # fname = '/home/tomas/Data/liver_segmentation/tryba/data_other/org-exp_183_46324212_venous_5.0_B30f-.pklz'

    # hypo in venous -----------------------
    # arterial - bad
    # fname = '/home/tomas/Data/liver_segmentation_06mm/tryba/data_other/org-exp_186_49290986_venous_0.6_B20f-.pklz'
    # venous - good
    fname = '/home/tomas/Data/liver_segmentation_06mm/tryba/data_other/org-exp_186_49290986_arterial_0.6_B20f-.pklz'

    # hyper, 1 on the border -------------------
    # arterial 0.6mm - not that bad
    # fname = '/home/tomas/Data/liver_segmentation_06mm/hyperdenzni/org-exp_239_61293268_DE_Art_Abd_0.75_I26f_M_0.5-.pklz'
    # venous 5mm - bad
    # fname = '/home/tomas/Data/liver_segmentation_06mm/hyperdenzni/org-exp_239_61293268_DE_Ven_Abd_0.75_I26f_M_0.5-.pklz'

    # shluk -----------------
    # arterial 5mm
    # fname = '/home/tomas/Data/liver_segmentation/tryba/data_other/org-exp_180_49509315_arterial_5.0_B30f-.pklz'
    # fname = '/home/tomas/Data/liver_segmentation_06mm/tryba/data_other/org-exp_180_49509315_arterial_0.6_B20f-.pklz'

    # targeted
    # arterial 0.6mm - bad
    # fname = '/home/tomas/Data/liver_segmentation_06mm/hyperdenzni/org-exp_238_54280551_Abd_Arterial_0.75_I26f_3-.pklz'
    # venous 0.6mm - b  ad
    # fname = '/home/tomas/Data/liver_segmentation_06mm/hyperdenzni/org-exp_238_54280551_Abd_Venous_0.75_I26f_3-.pklz'

    # parameters --------------------------------
    params = dict()
    # params['hypo_label'] = 1
    # params['hyper_label'] = 2
    healthy_label = 0
    hypo_label = 1
    hyper_label = 2

    # preparing data -----------------------------
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

    # runing application -------------------------
    run(im, labels, healthy_label, hypo_label, hyper_label)

    # app = QtGui.QApplication(sys.argv)
    # myapp = Lession_editor(im, labels, healthy_label, hypo_label, hyper_label)
    #
    # # zviditelneni aplikace
    # myapp.show()
    # sys.exit(app.exec_())