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
import Viewer_3D

from sklearn import metrics
from sklearn.cluster import KMeans

import pickle

import ConfigParser

import logging
logger = logging.getLogger(__name__)
logging.basicConfig()


from lession_editor_GUI import Ui_MainWindow
import Form_widget
import Hist_widget
import My_table_model as mtm

import Computational_core

import data_view_widget

class Lession_editor(QtGui.QMainWindow):
    """Main class of the programm."""

    def __init__(self, fname, disp_smoothed=False, parent=None):
        QtGui.QWidget.__init__(self, parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # uprava stylu pro lepsi vizualizaci splitteru
        QtGui.QApplication.setStyle(QtGui.QStyleFactory.create('Cleanlooks'))

        # self.im = im
        # self.labels = labels
        self.show_view_1 = True
        self.show_view_2 = False
        # self.healthy_label = healthy_label
        # self.hypo_label = hypo_label
        # self.hyper_label = hyper_label
        self.disp_smoothed = disp_smoothed
        self.view_L_curr_idx = 0
        self.view_R_curr_idx = 0

        # load parameters
        self.params = self.load_parameters()
        self.win_l = self.params['win_level']
        self.win_w = self.params['win_width']

        # fill parameters to widgets
        self.fill_parameters()

        self.voxel_size = self.params['voxel_size']

        # computational core
        self.cc = Computational_core.Computational_core(fname, self.params, self.statusBar())
        if self.cc.data_1.loaded:
            self.ui.serie_1_RB.setText('Serie #1: ' + self.cc.data_1.filename.split('/')[-1])
            self.ui.figure_1_CB.addItem(self.cc.data_1.filename.split('/')[-1])
            self.ui.figure_2_CB.addItem(self.cc.data_1.filename.split('/')[-1])
        if self.cc.data_2.loaded:
            self.ui.serie_2_RB.setText('Serie #2: ' + self.cc.data_2.filename.split('/')[-1])
            self.ui.figure_1_CB.addItem(self.cc.data_2.filename.split('/')[-1])
            self.ui.figure_2_CB.addItem(self.cc.data_2.filename.split('/')[-1])

        # radio buttons
        self.ui.serie_1_RB.clicked.connect(self.serie_1_RB_callback)
        self.ui.serie_2_RB.clicked.connect(self.serie_2_RB_callback)
        if self.ui.serie_1_RB.isChecked():
            self.cc.active_serie = 1
        else:
            self.cc.active_serie = 2

        self.ui.action_Load_serie_1.triggered.connect(lambda: self.action_Load_serie_callback(1))
        self.ui.action_Load_serie_2.triggered.connect(lambda: self.action_Load_serie_callback(2))

        # self.n_slices = self.data.shape[0]
        # self.n_slices = self.cc.data_1.n_slices

        # seting up the callback for the test button --------------------------------------
        # self.ui.test_BTN.clicked.connect(self.test_callback)
        #----------------------------------------------------------------------------------

        # seting up the range of the scrollbar to cope with the number of slices
        if self.cc.active_serie == 1:
            self.ui.slice_scrollB.setMaximum(self.cc.data_1.n_slices - 1)
            self.ui.slice_1_SB.setMaximum(self.cc.data_1.n_slices - 1)
        else:
            self.ui.slice_scrollB.setMaximum(self.cc.data_2.n_slices - 1)
            self.ui.slice_1_SB.setMaximum(self.cc.data_2.n_slices - 1)

        if self.cc.data_2.loaded:
            self.ui.slice_2_SB.setMaximum(self.cc.data_2.n_slices - 1)

        # combo boxes - for figure views
        self.ui.figure_1_CB.currentIndexChanged.connect(self.figure_1_CB_callback)
        self.ui.figure_2_CB.currentIndexChanged.connect(self.figure_2_CB_callback)

        # adding widget for displaying image data
        # self.form_widget = Form_widget.Form_widget(self, self.cc)
        if self.cc.actual_data.loaded:
            shape = self.cc.actual_data.shape
        else:
            shape = (0,0)
        height = 500
        vscale = self.voxel_size / float(np.min(self.voxel_size))
        grid = height / float(shape[1] * vscale[1])
        mgrid = (grid * vscale[0], grid * vscale[1])
        self.view_L = data_view_widget.SliceBox(shape[1:], mgrid)
        self.view_L.setCW(self.win_l, 'c')
        self.view_L.setCW(self.win_w, 'w')
        self.view_L.setSlice(self.cc.actual_data.data[0,:,:])
        data_viewer_layout = QtGui.QHBoxLayout()
        # data_viewer_layout.addWidget(self.form_widget)
        data_viewer_layout.addWidget(self.view_L)
        self.ui.viewer_F.setLayout(data_viewer_layout)

        # adding widget for displaying data histograms
        self.hist_widget = Hist_widget.Hist_widget(self, self.cc)
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
        self.ui.run_BTN.clicked.connect(self.run_callback)

        # connecting spin boxes
        self.ui.hypo_mean_SB.valueChanged.connect(self.hypo_mean_SB_callback)
        self.ui.hypo_std_SB.valueChanged.connect(self.hypo_std_SB_callback)
        self.ui.hyper_mean_SB.valueChanged.connect(self.hyper_mean_SB_callback)
        self.ui.hyper_std_SB.valueChanged.connect(self.hyper_std_SB_callback)
        self.ui.heal_mean_SB.valueChanged.connect(self.heal_mean_SB_callback)
        self.ui.heal_std_SB.valueChanged.connect(self.heal_std_SB_callback)

        # connecting scrollbars
        self.ui.slice_scrollB.valueChanged.connect(self.slider_changed)
        self.ui.slice_1_SB.valueChanged.connect(self.slider_1_changed)
        self.ui.slice_2_SB.valueChanged.connect(self.slider_2_changed)

        # connecting sliders with their line edit
        self.connect_SL_and_LE()


    def test_callback(self):
        n_rows, n_cols, n_slices = self.cc.labels.shape
        self.cc.labels = np.zeros(self.labels.shape)
        self.cc.labels[n_rows/3.:2*n_rows/3., n_cols/3.:2*n_cols/3.] = self.hypo


    def serie_1_RB_callback(self):
        self.cc.active_serie = 1
        self.cc.actual_data = self.cc.data_1


    def serie_2_RB_callback(self):
        self.cc.active_serie = 2
        self.cc.actual_data = self.cc.data_2


    def connect_SL_and_LE(self):
        # win width
        self.ui.win_wdth_SL.valueChanged.connect(self.win_wdth_SL_changed)
        ww_val = QtGui.QIntValidator(self.ui.win_wdth_SL.minimum(), self.ui.win_wdth_SL.maximum())
        self.ui.win_width_LE.setValidator(ww_val)
        self.ui.win_width_LE.textChanged.connect(self.win_width_LE_changed)

        # win level
        self.ui.win_lvl_SL.valueChanged.connect(self.win_lvl_SL_changed)
        wl_val = QtGui.QIntValidator(self.ui.win_lvl_SL.minimum(), self.ui.win_lvl_SL.maximum())
        self.ui.win_level_LE.setValidator(wl_val)
        self.ui.win_level_LE.textChanged.connect(self.win_level_LE_changed)

        # voxel size
        self.ui.voxel_size_SB.valueChanged.connect(self.voxel_size_SB_changed)
        vs_val = QtGui.QIntValidator(self.ui.voxel_size_SB.minimum(), self.ui.voxel_size_SB.maximum())
        self.ui.voxel_size_LE.setValidator(vs_val)
        self.ui.voxel_size_LE.textChanged.connect(self.voxel_size_LE_changed)

        # smoothing - sigma
        self.ui.sigma_SL.valueChanged.connect(self.sigma_SL_changed)
        ss_val = QtGui.QIntValidator(self.ui.sigma_SL.minimum(), self.ui.sigma_SL.maximum())
        self.ui.gaussian_sigma_LE.setValidator(ss_val)
        self.ui.gaussian_sigma_LE.textChanged.connect(self.gaussian_sigma_LE_changed)

        # smoothing - sigma_range
        self.ui.sigma_range_SL.valueChanged.connect(self.sigma_range_SL_changed)
        sr_val = QtGui.QIntValidator(self.ui.sigma_range_SL.minimum(), self.ui.sigma_range_SL.maximum())
        self.ui.bilateral_range_LE.setValidator(sr_val)
        self.ui.bilateral_range_LE.textChanged.connect(self.bilateral_range_LE_changed)

        # smoothing - sigma_spatial
        self.ui.sigma_spatial_SL.valueChanged.connect(self.sigma_spatial_SL_changed)
        sigs_val = QtGui.QIntValidator(self.ui.sigma_spatial_SL.minimum(), self.ui.sigma_spatial_SL.maximum())
        self.ui.bilateral_spatial_LE.setValidator(sigs_val)
        self.ui.bilateral_spatial_LE.textChanged.connect(self.bilateral_spatial_LE_changed)

        # smoothing - tv_weight
        self.ui.tv_weight_SL.valueChanged.connect(self.tv_weight_SL_changed)
        tvw_val = QtGui.QIntValidator(self.ui.tv_weight_SL.minimum(), self.ui.tv_weight_SL.maximum())
        self.ui.tv_weight_LE.setValidator(tvw_val)
        self.ui.tv_weight_LE.textChanged.connect(self.tv_weight_LE_changed)

        # alpha
        self.ui.alpha_SL.valueChanged.connect(self.alpha_SL_changed)
        alpha_val = QtGui.QIntValidator(self.ui.alpha_SL.minimum(), self.ui.alpha_SL.maximum())
        self.ui.alpha_LE.setValidator(alpha_val)
        self.ui.alpha_LE.textChanged.connect(self.alpha_LE_changed)

        # beta
        self.ui.beta_SL.valueChanged.connect(self.beta_SL_changed)
        beta_val = QtGui.QIntValidator(self.ui.beta_SL.minimum(), self.ui.beta_SL.maximum())
        self.ui.beta_LE.setValidator(beta_val)
        self.ui.beta_LE.textChanged.connect(self.beta_LE_changed)

        # frac
        self.ui.frac_SL.valueChanged.connect(self.frac_SL_changed)
        frac_val = QtGui.QIntValidator(self.ui.frac_SL.minimum(), self.ui.frac_SL.maximum())
        self.ui.perc_LE.setValidator(frac_val)
        self.ui.perc_LE.textChanged.connect(self.perc_LE_changed)

        # heal_std_k
        self.ui.heal_std_k_SL.valueChanged.connect(self.heal_std_k_SL_changed)
        stdh_val = QtGui.QIntValidator(self.ui.heal_std_k_SL.minimum(), self.ui.heal_std_k_SL.maximum())
        self.ui.k_std_h_LE.setValidator(stdh_val)
        self.ui.k_std_h_LE.textChanged.connect(self.k_std_h_LE_changed)

        # tum_std_k
        self.ui.tum_std_k_SL.valueChanged.connect(self.tum_std_k_SL_changed)
        stdt_val = QtGui.QIntValidator(self.ui.tum_std_k_SL.minimum(), self.ui.tum_std_k_SL.maximum())
        self.ui.k_std_t_LE.setValidator(stdt_val)
        self.ui.k_std_t_LE.textChanged.connect(self.k_std_t_LE_changed)

        # min_area
        self.ui.min_area_SL.valueChanged.connect(self.min_area_SL_changed)
        minarea_val = QtGui.QIntValidator(self.ui.min_area_SL.minimum(), self.ui.min_area_SL.maximum())
        self.ui.min_area_LE.setValidator(minarea_val)
        self.ui.min_area_LE.textChanged.connect(self.min_area_LE_changed)

        # max_area
        self.ui.max_area_SL.valueChanged.connect(self.max_area_SL_changed)
        maxarea_val = QtGui.QIntValidator(self.ui.max_area_SL.maximum(), self.ui.max_area_SL.maximum())
        self.ui.max_area_LE.setValidator(maxarea_val)
        self.ui.max_area_LE.textChanged.connect(self.max_area_LE_changed)

        # min_comp
        self.ui.min_comp_SL.valueChanged.connect(self.min_comp_SL_changed)
        mincomp_val = QtGui.QIntValidator(self.ui.min_comp_SL.maximum(), self.ui.min_comp_SL.maximum())
        self.ui.min_comp_LE.setValidator(mincomp_val)
        self.ui.min_comp_LE.textChanged.connect(self.min_comp_LE_changed)

        # comp_fact
        self.ui.comp_fact_SB.valueChanged.connect(self.comp_fact_SB_changed)
        compf_val = QtGui.QIntValidator(self.ui.comp_fact_SB.maximum(), self.ui.comp_fact_SB.maximum())
        self.ui.comp_fact_LE.setValidator(compf_val)
        self.ui.comp_fact_LE.textChanged.connect(self.comp_fact_LE_changed)

        # tableview selection changed
        # self.ui.objects_TV.selectionModel().selectionChanged.connect(self.selection_changed)


    def selection_changed(self):
        indexes = self.ui.objects_TV.selectionModel().selectedRows()
        for index in indexes:
            print index.row()


    def comp_fact_SB_changed(self, value):
        self.ui.comp_fact_LE.setText(str(value))
        self.params['comp_fact'] = value

    def comp_fact_LE_changed(self, value):
        try:  # must be due to the possibility that no character could be entered
            self.ui.comp_fact_SB.setValue(int(value))
            self.params['comp_fact'] = int(value)
        except:
            pass

    def min_comp_SL_changed(self, value):
        self.ui.min_comp_LE.setText(str(value))
        self.params['min_compactness'] = value
        self.cc.objects_filtration()
        self.fill_table(self.cc.labels[self.cc.filtered_idxs], self.cc.areas[self.cc.filtered_idxs], self.cc.comps[self.cc.filtered_idxs])

    def min_comp_LE_changed(self, value):
        try:  # must be due to the possibility that no character could be entered
            self.ui.min_comp_SL.setValue(int(value))
            self.params['min_compactness'] = int(value)
            self.cc.objects_filtration()
            self.fill_table(self.cc.labels[self.cc.filtered_idxs], self.cc.areas[self.cc.filtered_idxs], self.cc.comps[self.cc.filtered_idxs])
        except:
            pass

    def max_area_SL_changed(self, value):
        self.ui.max_area_LE.setText(str(value))
        self.params['max_area'] = value
        self.cc.objects_filtration()
        self.fill_table(self.cc.labels[self.cc.filtered_idxs], self.cc.areas[self.cc.filtered_idxs], self.cc.comps[self.cc.filtered_idxs])

    def max_area_LE_changed(self, value):
        try:  # must be due to the possibility that no character could be entered
            self.ui.max_area_SL.setValue(int(value))
            self.params['max_area'] = int(value)
            self.cc.objects_filtration()
            self.fill_table(self.cc.labels[self.cc.filtered_idxs], self.cc.areas[self.cc.filtered_idxs], self.cc.comps[self.cc.filtered_idxs])
        except:
            pass

    def min_area_SL_changed(self, value):
        self.ui.min_area_LE.setText(str(value))
        self.params['min_area'] = value
        self.cc.objects_filtration()
        self.fill_table(self.cc.labels[self.cc.filtered_idxs], self.cc.areas[self.cc.filtered_idxs], self.cc.comps[self.cc.filtered_idxs])


    def min_area_LE_changed(self, value):
        try:  # must be due to the possibility that no character could be entered
            self.ui.min_area_SL.setValue(int(value))
            self.params['min_area'] = int(value)
            self.cc.objects_filtration()
            self.fill_table(self.cc.labels[self.cc.filtered_idxs], self.cc.areas[self.cc.filtered_idxs], self.cc.comps[self.cc.filtered_idxs])
        except:
            pass

    def tum_std_k_SL_changed(self, value):
        self.ui.k_std_t_LE.setText(str(value))
        self.params['k_std_t'] = value

    def k_std_t_LE_changed(self, value):
        try:  # must be due to the possibility that no character could be entered
            self.ui.tum_std_k_SL.setValue(int(value))
            self.params['k_std_t'] = int(value)
        except:
            pass

    def heal_std_k_SL_changed(self, value):
        self.ui.k_std_h_LE.setText(str(value))
        self.params['k_std_h'] = value

    def k_std_h_LE_changed(self, value):
        try:  # must be due to the possibility that no character could be entered
            self.ui.heal_std_k_SL.setValue(int(value))
            self.params['k_std_h'] = int(value)
        except:
            pass

    def frac_SL_changed(self, value):
        self.ui.perc_LE.setText(str(value))
        self.params['perc'] = value

    def perc_LE_changed(self, value):
        try:  # must be due to the possibility that no character could be entered
            self.ui.frac_SL.setValue(int(value))
            self.params['perc'] = int(value)
        except:
            pass

    def beta_SL_changed(self, value):
        self.ui.beta_LE.setText(str(value))
        self.params['beta'] = value

    def beta_LE_changed(self, value):
        try:  # must be due to the possibility that no character could be entered
            self.ui.beta_SL.setValue(int(value))
            self.params['beta'] = int(value)
        except:
            pass

    def alpha_SL_changed(self, value):
        self.ui.alpha_LE.setText(str(value))
        self.params['alpha'] = value

    def alpha_LE_changed(self, value):
        try:  # must be due to the possibility that no character could be entered
            self.ui.alpha_SL.setValue(int(value))
            self.params['alpha'] = int(value)
        except:
            pass

    def tv_weight_SL_changed(self, value):
        self.ui.tv_weight_LE.setText(str(float(value)/1000))
        self.params['tv_weight'] = value

    def tv_weight_LE_changed(self, value):
        try:  # must be due to the possibility that no character could be entered
            w = int(float(value)*1000)
            self.ui.tv_weight_SL.setValue(w)
            self.params['tv_weight'] = w
        except:
            pass

    def sigma_spatial_SL_changed(self, value):
        self.ui.bilateral_spatial_LE.setText(str(value))
        self.params['sigma_spatial'] = value

    def bilateral_spatial_LE_changed(self, value):
        try:  # must be due to the possibility that no character could be entered
            self.ui.sigma_spatial_SL.setValue(int(value))
            self.params['sigma_spatial'] = int(value)
        except:
            pass

    def sigma_range_SL_changed(self, value):
        self.ui.bilateral_range_LE.setText(str(value))
        self.params['sigma_range'] = value

    def bilateral_range_LE_changed(self, value):
        try:  # must be due to the possibility that no character could be entered
            self.ui.sigma_range_SL.setValue(int(value))
            self.params['sigma_range'] = int(value)
        except:
            pass

    def sigma_SL_changed(self, value):
        self.ui.gaussian_sigma_LE.setText(str(value))
        self.params['sigma'] = value

    def gaussian_sigma_LE_changed(self, value):
        try:  # must be due to the possibility that no character could be entered
            self.ui.sigma_SL.setValue(int(value))
            self.params['sigma'] = int(value)
        except:
            pass

    def voxel_size_SB_changed(self, value):
        self.ui.voxel_size_LE.setText(str(value))
        self.params['working_voxel_size'] = value

    def voxel_size_LE_changed(self, value):
        try:  # must be due to the possibility that no character could be entered
            self.ui.voxel_size_SB.setValue(int(value))
            self.params['working_voxel_size'] = int(value)
        except:
            pass

    def win_wdth_SL_changed(self, value):
        self.ui.win_width_LE.setText(str(value))
        self.params['win_width'] = value

    def win_width_LE_changed(self, value):
        try:  # must be due to the possibility that no character could be entered
            self.ui.win_wdth_SL.setValue(int(value))
            self.params['win_width'] = int(value)
        except:
            pass

    def win_lvl_SL_changed(self, value):
        self.ui.win_level_LE.setText(str(value))
        self.params['win_level'] = value

    def win_level_LE_changed(self, value):
        try:  # must be due to the possibility that the character '-' or nothing could be entered
            self.ui.win_lvl_SL.setValue(int(value))
            self.params['win_level'] = int(value)
        except:
            pass


    def load_parameters(self, config_path='config.xml'):
        config = ConfigParser.ConfigParser()
        config.read('config.ini')

        params = dict()

        # an automatic way
        for section in config.sections():
            for option in config.options(section):
                try:
                    params[option] = config.getint(section, option)
                except ValueError:
                    try:
                        params[option] = config.getfloat(section, option)
                    except ValueError:
                        if option == 'voxel_size':
                            str = config.get(section, option)
                            params[option] = np.array(map(int, str.split(', ')))
                        else:
                            params[option] = config.get(section, option)

        return params


    def fill_parameters(self):
        # general parameters
        self.ui.win_wdth_SL.setValue(self.params['win_width'])
        self.ui.win_width_LE.setText(str(self.params['win_width']))

        self.ui.win_lvl_SL.setValue(self.params['win_level'])
        self.ui.win_level_LE.setText(str(self.params['win_level']))

        self.ui.voxel_size_SB.setValue(self.params['working_voxel_size_mm'])
        self.ui.voxel_size_LE.setText(str(self.params['working_voxel_size_mm']))

        # smoothing parameters
        self.ui.sigma_SL.setValue(self.params['sigma'])
        self.ui.gaussian_sigma_LE.setText(str(self.params['sigma']))

        self.ui.sigma_range_SL.setValue(self.params['sigma_range'])
        self.ui.bilateral_range_LE.setText(str(self.params['sigma_range']))

        self.ui.sigma_spatial_SL.setValue(self.params['sigma_spatial'])
        self.ui.bilateral_spatial_LE.setText(str(self.params['sigma_spatial']))

        self.ui.tv_weight_SL.setValue(self.params['tv_weight'])
        self.ui.tv_weight_LE.setText(str(self.params['tv_weight']))

        # color model parameters
        self.ui.frac_SL.setValue(self.params['perc'])
        self.ui.perc_LE.setText(str(self.params['perc']))

        self.ui.heal_std_k_SL.setValue(self.params['k_std_h'])
        self.ui.k_std_h_LE.setText(str(self.params['k_std_h']))

        self.ui.tum_std_k_SL.setValue(self.params['k_std_t'])
        self.ui.k_std_t_LE.setText(str(self.params['k_std_t']))

        # localization parameters
        self.ui.alpha_SL.setValue(self.params['alpha'])
        self.ui.alpha_LE.setText(str(self.params['alpha']))

        self.ui.beta_SL.setValue(self.params['beta'])
        self.ui.beta_LE.setText(str(self.params['beta']))

        self.ui.min_area_SL.setValue(self.params['min_area'])
        self.ui.min_area_LE.setText(str(self.params['min_area']))

        self.ui.max_area_SL.setValue(self.params['max_area'])
        self.ui.max_area_LE.setText(str(self.params['max_area']))

        self.ui.min_comp_SL.setValue(self.params['min_compactness'])
        self.ui.min_comp_LE.setText(str(self.params['min_compactness']))

        self.ui.comp_fact_SB.setValue(self.params['comp_fact'])
        self.ui.comp_fact_LE.setText(str(self.params['comp_fact']))



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


    # def wheelEvent(self, event):
    #     print event.delta() / 120


    def scroll_event(self, value, who):
        if who == 0:  # left viewer
            new = self.actual_slice_L + value
            if (new < 0) or (new >= self.data_L.n_slices):
                return
        elif who == 1:  # right viewer
            new = self.actual_slice_R + value
            if (new < 0) or (new >= self.data_R.n_slices):
                return



    def slider_changed(self, val):
        # if val > self.view_L_curr_idx:
        #     self.form_widget.scroll_next()
        # else:
        #     self.form_widget.scroll_prev()
        # self.slice_change(val)
        # self.slice_1_change(val)
        if (val > 0) and (val < self.data_L.n_slices):
            self.actual_slice_L = val
        else:
            return

        if val < self.data_R.n_slices:
            self.actual_slice_R = val
        else:
            self.actual_slice_R = self.data_R.n_slices




    def slice_change(self, val):
        self.ui.slice_scrollB.setValue(val)
        self.ui.slice_number_LBL.setText('slice # = %i' % (val + 1))


    def slider_1_changed(self, val):
        pass
        # self.view_1_curr_idx = val
        # if val > self.view_L_curr_idx:
        #     self.form_widget.scroll_next()
        # else:
        #     self.form_widget.scroll_prev()
        # self.slice_1_change(val)
        # self.slice_2_change(val)
        # self.slice_change(val)


    def slider_2_changed(self, val):
        pass
        # self.view_2_curr_idx = val
        # # self.form_widget.actual_slice_2 = val
        # self.slice_2_change(val)
        # self.form_widget.update_figures()


    def slice_1_change(self, val):
        self.ui.slice_1_SB.setValue(val)
        self.ui.slice_number_1_LBL.setText(str(val + 1))


    def slice_2_change(self, val):
        self.ui.slice_2_SB.setValue(val)
        self.ui.slice_number_2_LBL.setText(str(val + 1))


    def calculate_models_callback(self):
        self.statusBar().showMessage('Calculating intensity models...')
        self.cc.calculate_intensity_models()
        self.statusBar().showMessage('Intensity models calculated.')
        self.update_models()


    def update_models(self):
        self.hist_widget.update_heal_rv(self.cc.models['rv_heal'])
        self.hist_widget.update_hypo_rv(self.cc.models['rv_hypo'])
        self.hist_widget.update_hyper_rv(self.cc.models['rv_hyper'])

        self.ui.heal_mean_SB.setValue(self.cc.models['rv_heal'].mean())
        self.ui.heal_std_SB.setValue(self.cc.models['rv_heal'].std())

        self.ui.hypo_mean_SB.setValue(self.cc.models['rv_hypo'].mean())
        self.ui.hypo_std_SB.setValue(self.cc.models['rv_hypo'].std())

        self.ui.hyper_mean_SB.setValue(self.cc.models['rv_hyper'].mean())
        self.ui.hyper_std_SB.setValue(self.cc.models['rv_hyper'].std())

        self.hist_widget.update_figures()


    def run_callback(self):
        # Viewer_3D.run(self.data)
        # viewer = Viewer_3D.Viewer_3D(self.data)
        # viewer.show()

        # run localization
        self.statusBar().showMessage('Localization started...')
        self.cc.run()
        self.update_models()

        self.form_widget.update_figures()
        # self.hist_widget.update_figures()
        self.labels = self.cc.actual_data.labels
        # self.labels = self.cc.labels

        # filling table with objects
        self.fill_table(self.cc.labels[self.cc.filtered_idxs], self.cc.areas[self.cc.filtered_idxs], self.cc.comps[self.cc.filtered_idxs])

        # self.cc.areas = np.array([10, 20, 30, 8])
        # self.cc.comps = np.array([51, 60, 70, 80])
        # self.cc.objects = np.array([[0,0,-1,3,3],[0,-1,-1,3,3],[-1,-1,-1,-1,-1],[1,1,-1,-1,2],[1,-1,-1,2,2]])
        # self.cc.labels = np.unique(self.cc.objects)[1:]
        # self.cc.n_objects = len(np.unique(self.cc.objects)) - 1
        # self.fill_table(self.cc.labels, self.cc.areas, self.cc.comps)


    def fill_table(self, labels, areas, comps):
        # idxs = np.arange(1, len(areas)+1)
        features = np.vstack((labels, areas, comps)).T
        self.table_model = mtm.MyTableModel(features)
        self.ui.objects_TV.setModel(self.table_model)
        self.ui.objects_TV.selectionModel().selectionChanged.connect(self.selection_changed)
        # self.tableview.verticalHeader().setVisible(True)


    # def update_table_model(self):
    #     self.ui.objects_TV


    def view_1_callback(self):
        self.show_view_1 = not self.show_view_1

        # enabling and disabling other toolbar icons
        self.ui.show_im_1_BTN.setEnabled(not self.ui.show_im_1_BTN.isEnabled())
        self.ui.show_labels_1_BTN.setEnabled(not self.ui.show_labels_1_BTN.isEnabled())
        self.ui.show_contours_1_BTN.setEnabled(not self.ui.show_contours_1_BTN.isEnabled())

        self.statusBar().showMessage('view_1 set to %s' % self.show_view_1)
        # print 'view_1 set to', self.show_view_1

        self.form_widget.update_figures()


    def view_2_callback(self):
        self.show_view_2 = not self.show_view_2

        # enabling and disabling other toolbar icons
        self.ui.show_im_2_BTN.setEnabled(not self.ui.show_im_2_BTN.isEnabled())
        self.ui.show_labels_2_BTN.setEnabled(not self.ui.show_labels_2_BTN.isEnabled())
        self.ui.show_contours_2_BTN.setEnabled(not self.ui.show_contours_2_BTN.isEnabled())

        self.statusBar().showMessage('view_2 set to %s' % self.show_view_2)
        # print 'view_2 set to', self.show_view_2

        self.form_widget.update_figures()


    def show_im_1_callback(self):
        # print 'data_1 set to im'
        self.statusBar().showMessage('data_1 set to im')
        # self.form_widget.data_1 = self.data
        self.form_widget.data_L_str = 'im'
        self.form_widget.update_figures()


    def show_im_2_callback(self):
        # print 'data_2 set to im'
        self.statusBar().showMessage('data_2 set to im')
        # if self.disp_smoothed:
        #     self.form_widget.data_2 = self.labels
        # else:
        #     self.form_widget.data_2 = self.data
        self.form_widget.data_R_str = 'im'
        self.form_widget.update_figures()


    def show_labels_1_callback(self):
        # print 'data_1 set to labels'
        self.statusBar().showMessage('data_1 set to labels')
        # self.form_widget.data_1 = self.labels
        self.form_widget.data_L_str = 'labels'
        self.form_widget.update_figures()


    def show_labels_2_callback(self):
        # print 'data_2 set to labels'
        self.statusBar().showMessage('data_2 set to labels')
        # self.form_widget.data_1 = self.labels
        self.form_widget.data_R_str = 'labels'
        self.form_widget.update_figures()


    def show_contours_1_callback(self):
        # print 'data_2 set to contours'
        self.statusBar().showMessage('data_1 set to contours')
        # self.form_widget.data_1 = self.data
        self.form_widget.data_L_str = 'contours'
        self.form_widget.update_figures()


    def show_contours_2_callback(self):
        # print 'data_2 set to contours'
        self.statusBar().showMessage('data_2 set to contours')
        # self.form_widget.data_2 = self.data
        self.form_widget.data_R_str = 'contours'
        self.form_widget.update_figures()


    def figure_1_CB_callback(self):
        if self.ui.figure_1_CB.currentIndex() == 0:
            self.data_L = self.cc.data_1
            # self.ui.slice_1_SB.setMaximum(self.cc.data_1.n_slices)
        elif self.ui.figure_1_CB.currentIndex() == 1:
            self.data_L = self.cc.data_2

        if self.actual_slice_L >= self.data_L.n_slices:
            self.actual_slice_L = self.data_L.n_slices - 1
            self.slice_change(self.actual_slice_L)

        self.ui.slice_1_SB.setMaximum(self.data_L.n_slices - 1)
        self.ui.slice_scrollB.setMaximum(self.data_L.n_slices - 1)

        self.form_widget.update_figures()

    def figure_2_CB_callback(self):
        if self.ui.figure_2_CB.currentIndex() == 0:
            self.data_R = self.cc.data_1
            # self.ui.slice_2_SB.setMaximum(self.cc.data_1.n_slices)
        elif self.ui.figure_2_CB.currentIndex() == 1:
            self.data_R = self.cc.data_2
            # self.ui.slice_1_SB.setMaximum(self.cc.data_2.n_slices)

        if self.actual_slice_R >= self.data_R.n_slices:
            self.actual_slice_R = self.data_R.n_slices - 1
            self.slice_change(self.actual_slice_R)

        self.ui.slice_2_SB.setMaximum(self.data_R.n_slices - 1)
        # self.ui.slice_scrollB.setMaximum(self.data_R.n_slices - 1)

        self.form_widget.update_figures()


    def action_Load_serie_callback(self, serie_number):
        fname = QtGui.QFileDialog.getOpenFileName(self, 'Open file', self.params['data_dir'])
        print 'Does not work yet.'
        print fname


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
    fnames = list()
    fnames.append('/home/tomas/Data/liver_segmentation/tryba/data_other/org-exp_183_46324212_venous_5.0_B30f-.pklz')
    fnames.append('/home/tomas/Data/liver_segmentation/tryba/data_other/org-exp_183_46324212_arterial_5.0_B30f-.pklz')

    # hypo in venous -----------------------
    # arterial - bad
    # fname = '/home/tomas/Data/liver_segmentation_06mm/tryba/data_other/org-exp_186_49290986_venous_0.6_B20f-.pklz'
    # venous - good
    # fname = '/home/tomas/Data/liver_segmentation_06mm/tryba/data_other/org-exp_186_49290986_arterial_0.6_B20f-.pklz'

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
    # venous 0.6mm - bad
    # fname = '/home/tomas/Data/liver_segmentation_06mm/hyperdenzni/org-exp_238_54280551_Abd_Venous_0.75_I26f_3-.pklz'

    # runing application -------------------------
    app = QtGui.QApplication(sys.argv)
    le = Lession_editor(fnames)
    le.show()
    sys.exit(app.exec_())

    # app = QtGui.QApplication(sys.argv)
    # myapp = Lession_editor(im, labels, healthy_label, hypo_label, hyper_label)
    #
    # # zviditelneni aplikace
    # myapp.show()
    # sys.exit(app.exec_())