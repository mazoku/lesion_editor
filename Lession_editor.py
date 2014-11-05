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

# import cv2
# import pygco

import tools
import py3DSeedEditor
# from mayavi import mlab

import TumorVisualiser

from sklearn import metrics
from sklearn.cluster import KMeans

import pickle

import logging
logger = logging.getLogger(__name__)
logging.basicConfig()


from lession_editor_GUI import Ui_MainWindow
import Form_widget
import Hist_widget

import Computational_core

class Lession_editor(QtGui.QMainWindow):
    """Main class of the programm."""

    def __init__(self, fname, healthy_label=0, hypo_label=1, hyper_label=2, disp_smoothed=False, parent=None):
        QtGui.QWidget.__init__(self, parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # uprava stylu pro lepsi vizualizaci splitteru
        QtGui.QApplication.setStyle(QtGui.QStyleFactory.create('Cleanlooks'))

        # self.im = im
        # self.labels = labels
        self.show_view_1 = True
        self.show_view_2 = True
        self.healthy_label = healthy_label
        self.hypo_label = hypo_label
        self.hyper_label = hyper_label
        self.disp_smoothed = disp_smoothed

        # computational core
        self.cc = Computational_core.Computational_core(fname)

        self.data = self.cc.data
        self.labels = np.zeros(self.data.shape, dtype=np.int)
        self.mask = self.cc.mask

        self.n_slices = self.data.shape[0]

        # nastaveni rozsahu scrollBaru podle poctu rezu
        self.ui.slice_scrollB.setMaximum(self.n_slices - 1)

        # adding widget for displaying image data
        self.form_widget = Form_widget.Form_widget(self)
        data_viewer_layout = QtGui.QHBoxLayout()
        data_viewer_layout.addWidget(self.form_widget)
        self.ui.viewer_F.setLayout(data_viewer_layout)

        # adding widget for displaying data histograms
        self.hist_widget = Hist_widget.Hist_widget(self)
        hist_viewer_layout = QtGui.QHBoxLayout()
        hist_viewer_layout.addWidget(self.hist_widget)
        self.ui.histogram_F.setLayout(hist_viewer_layout)

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

        # main buttons
        self.ui.calculate_models_BTN.clicked.connect(self.calculate_models_callback)

        # connecting spin boxes
        self.ui.hypo_mean_SB.valueChanged.connect(self.hypo_mean_SB_callback)
        self.ui.hypo_std_SB.valueChanged.connect(self.hypo_std_SB_callback)
        self.ui.hyper_mean_SB.valueChanged.connect(self.hyper_mean_SB_callback)
        self.ui.hyper_std_SB.valueChanged.connect(self.hyper_std_SB_callback)
        self.ui.heal_mean_SB.valueChanged.connect(self.heal_mean_SB_callback)
        self.ui.heal_std_SB.valueChanged.connect(self.heal_std_SB_callback)


        # connecting slider
        self.ui.slice_scrollB.valueChanged.connect(self.slider_changed)


    def hypo_mean_SB_callback(self, value):
        self.statusBar().showMessage('Hypodense model updated thru spin box.')
        rv = scista.norm(value, self.cc.models['rv_hypo'].std())
        self.cc.models['rv_hypo'] = rv
        self.hist_widget.update_hypo_rv(rv)
        self.hist_widget.update_figures()


    def hypo_std_SB_callback(self, value):
        self.statusBar().showMessage('Hypodense model updated thru spin box.')
        rv = scista.norm(self.cc.models['rv_hypo'].mean(), value)
        self.cc.models['rv_hypo'] = rv
        self.hist_widget.update_hypo_rv(rv)
        self.hist_widget.update_figures()


    def hyper_mean_SB_callback(self, value):
        self.statusBar().showMessage('Hyperdense model updated thru spin box.')
        rv = scista.norm(value, self.cc.models['rv_hyper'].std())
        self.cc.models['rv_hyper'] = rv
        self.hist_widget.update_hyper_rv(rv)
        self.hist_widget.update_figures()


    def hyper_std_SB_callback(self, value):
        self.statusBar().showMessage('Hyperdense model updated thru spin box.')
        rv = scista.norm(self.cc.models['rv_hyper'].mean(), value)
        self.cc.models['rv_hyper'] = rv
        self.hist_widget.update_hyper_rv(rv)
        self.hist_widget.update_figures()


    def heal_mean_SB_callback(self, value):
        self.statusBar().showMessage('Healthy model updated thru spin box.')
        rv = scista.norm(value, self.cc.models['rv_heal'].std())
        self.cc.models['rv_heal'] = rv
        self.hist_widget.update_heal_rv(rv)
        self.hist_widget.update_figures()


    def heal_std_SB_callback(self, value):
        self.statusBar().showMessage('Healthy model updated thru spin box.')
        rv = scista.norm(self.cc.models['rv_heal'].mean(), value)
        self.cc.models['rv_heal'] = rv
        self.hist_widget.update_heal_rv(rv)
        self.hist_widget.update_figures()


    def wheelEvent(self, event):
        print event.delta() / 120


    def slider_changed(self, val):
        self.slice_change(val)
        self.form_widget.actual_slice = val
        self.form_widget.update_figures()


    def slice_change(self, val):
        self.ui.slice_scrollB.setValue(val)
        self.ui.slice_number_LBL.setText('slice # = %i' % (val + 1))


    def calculate_models_callback(self):
        self.statusBar().showMessage('Calculating intensity models...')

        self.cc.calculate_intensity_models()
        self.hist_widget.update_heal_rv(self.cc.models['rv_heal'])
        self.hist_widget.update_hypo_rv(self.cc.models['rv_hypo'])
        self.hist_widget.update_hyper_rv(self.cc.models['rv_hyper'])

        self.statusBar().showMessage('Intensity models calculated.')

        self.ui.heal_mean_SB.setValue(self.cc.models['rv_heal'].mean())
        self.ui.heal_std_SB.setValue(self.cc.models['rv_heal'].std())

        self.ui.hypo_mean_SB.setValue(self.cc.models['rv_hypo'].mean())
        self.ui.hypo_std_SB.setValue(self.cc.models['rv_hypo'].std())

        self.ui.hyper_mean_SB.setValue(self.cc.models['rv_hyper'].mean())
        self.ui.hyper_std_SB.setValue(self.cc.models['rv_hyper'].std())

        self.hist_widget.update_figures()


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


    def run(self, im, labels, healthy_label, hypo_label, hyper_label, slice_axis=2, disp_smoothed=False):
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
    # params = dict()
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
    # run(im, labels, healthy_label, hypo_label, hyper_label)
    app = QtGui.QApplication(sys.argv)
    le = Lession_editor(fname)
    le.show()
    sys.exit(app.exec_())

    # app = QtGui.QApplication(sys.argv)
    # myapp = Lession_editor(im, labels, healthy_label, hypo_label, hyper_label)
    #
    # # zviditelneni aplikace
    # myapp.show()
    # sys.exit(app.exec_())