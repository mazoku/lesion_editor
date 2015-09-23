from __future__ import division

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


from lession_editor_GUI_slim import Ui_MainWindow
import Form_widget
from hist_widget import Hist_widget
from objects_widget import Objects_widget
import My_table_model as mtm
import area_hist_widget as ahw
import Computational_core
import data_view_widget
import Lesion

# constants definition
SHOW_IM = 0
SHOW_LABELS = 1
SHOW_CONTOURS = 2
# SHOW_FILTERED_LABELS = 3

class LessionEditor(QtGui.QMainWindow):
    """Main class of the programm."""


    def __init__(self, fname, disp_smoothed=False, parent=None):
        QtGui.QWidget.__init__(self, parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # uprava stylu pro lepsi vizualizaci splitteru
        QtGui.QApplication.setStyle(QtGui.QStyleFactory.create('Cleanlooks'))

        # self.im = im
        # self.labels = labels
        self.show_view_L = True
        self.show_view_R = False
        # self.healthy_label = healthy_label
        # self.hypo_label = hypo_label
        # self.hyper_label = hyper_label
        self.disp_smoothed = disp_smoothed
        # self.view_L_curr_idx = 0
        # self.view_R_curr_idx = 0

        self.show_mode_L = SHOW_IM
        self.show_mode_R = SHOW_IM

        # load parameters
        self.params = self.load_parameters()
        self.win_l = self.params['win_level']
        self.win_w = self.params['win_width']

        self.data_L = None
        self.data_R = None
        self.actual_slice_L = 0
        self.actual_slice_R = 0

        self.selected_objects_labels = None  # list of objects' labels selected in tableview

        # fill parameters to widgets
        # self.fill_parameters()

        self.voxel_size = self.params['voxel_size']
        self.view_widget_width = 50
        self.two_views = False

        self.hist_widget = None

        # creating object widget, linking its sliders and buttons
        self.objects_widget = Objects_widget()
        self.params['compactness_step'] = self.objects_widget.ui.min_compactness_SL.singleStep() / self.objects_widget.ui.min_compactness_SL.maximum()
        self.objects_widget.area_RS.endValueChanged.connect(self.object_slider_changed)
        self.objects_widget.area_RS.startValueChanged.connect(self.object_slider_changed)
        self.objects_widget.density_RS.endValueChanged.connect(self.object_slider_changed)
        self.objects_widget.density_RS.startValueChanged.connect(self.object_slider_changed)
        self.objects_widget.ui.min_compactness_SL.valueChanged.connect(self.min_compactness_SL_changed)
        # connecting min_compactness line edit and slider
        ww_val = QtGui.QIntValidator(0, 1)
        self.objects_widget.ui.min_compactness_LE.setValidator(ww_val)
        self.objects_widget.ui.min_compactness_LE.textChanged.connect(self.min_compactness_LE_changed)
        # button callbacks
        self.objects_widget.ui.add_obj_BTN.clicked.connect(self.add_obj_BTN_callback)
        self.params['max_area'] = self.objects_widget.area_RS.end()
        self.params['min_area'] = self.objects_widget.area_RS.start()
        self.params['max_density'] = self.objects_widget.density_RS.end()
        self.params['min_density'] = self.objects_widget.density_RS.start()
        self.params['min_compactness'] = self.objects_widget.ui.min_compactness_SL.value() * self.params['compactness_step']

        # self.area_hist_widget = ahw.AreaHistWidget()

        # computational core
        self.cc = Computational_core.Computational_core(fname, self.params, self.statusBar())
        if self.cc.data_1.loaded:
            # self.ui.serie_1_RB.setText('Serie #1: ' + self.cc.data_1.filename.split('/')[-1])
            self.ui.figure_L_CB.addItem(self.cc.data_1.filename.split('/')[-1])
            self.ui.figure_R_CB.addItem(self.cc.data_1.filename.split('/')[-1])
            self.data_L = self.cc.data_1
            if not self.cc.data_2.loaded:
                self.data_R = self.cc.data_1
        if self.cc.data_2.loaded:
            # self.ui.serie_2_RB.setText('Serie #2: ' + self.cc.data_2.filename.split('/')[-1])
            self.ui.figure_L_CB.addItem(self.cc.data_2.filename.split('/')[-1])
            self.ui.figure_R_CB.addItem(self.cc.data_2.filename.split('/')[-1])
            self.data_R = self.cc.data_2
            self.ui.figure_R_CB.setCurrentIndex(1)
            if not self.cc.data_1.loaded:
                self.data_L = self.cc.data_2

        # radio buttons
        # self.ui.serie_1_RB.clicked.connect(self.serie_1_RB_callback)
        # self.ui.serie_2_RB.clicked.connect(self.serie_2_RB_callback)
        # if self.ui.serie_1_RB.isChecked():
        #     self.cc.active_serie = 1
        # else:
        #     self.cc.active_serie = 2
        self.cc.active_serie = 1

        self.ui.action_load_serie_1.triggered.connect(lambda: self.action_load_serie_callback(1))
        self.ui.action_load_serie_2.triggered.connect(lambda: self.action_load_serie_callback(2))

        self.ui.action_circle.triggered.connect(self.action_circle_callback)
        self.ui.action_show_color_model.triggered.connect(self.action_show_color_model_callback)
        self.ui.action_show_object_list.triggered.connect(self.action_show_object_list_callback)
        self.ui.action_run.triggered.connect(self.run_callback)

        # self.n_slices = self.data.shape[0]
        # self.n_slices = self.cc.data_1.n_slices

        # seting up the callback for the test button --------------------------------------
        # self.ui.test_BTN.clicked.connect(self.test_callback)
        #----------------------------------------------------------------------------------

        # seting up the range of the scrollbar to cope with the number of slices
        if self.cc.active_serie == 1:
            self.ui.slice_C_SB.setMaximum(self.cc.data_1.n_slices - 1)
            self.ui.slice_L_SB.setMaximum(self.cc.data_1.n_slices - 1)
        else:
            self.ui.slice_C_SB.setMaximum(self.cc.data_2.n_slices - 1)
            self.ui.slice_L_SB.setMaximum(self.cc.data_2.n_slices - 1)

        if self.cc.data_2.loaded:
            self.ui.slice_R_SB.setMaximum(self.cc.data_2.n_slices - 1)

        # combo boxes - for figure views
        self.ui.figure_L_CB.currentIndexChanged.connect(self.figure_L_CB_callback)
        self.ui.figure_R_CB.currentIndexChanged.connect(self.figure_R_CB_callback)

        # self.view_L = data_view_widget.SliceBox(self.data_L.shape[1:], self.voxel_size)
        self.view_L = data_view_widget.SliceBox(self.data_L.data_aview.shape[:-1], self.voxel_size, self)
        self.view_L.setCW(self.win_l, 'c')
        self.view_L.setCW(self.win_w, 'w')
        # self.view_L.setSlice(self.data_L.data[0,:,:])
        self.view_L.setSlice(self.data_L.data_aview[...,0])
        self.view_L.mouseClickSignal.connect(self.add_obj_event)
        self.view_L.mousePressEvent = self.view_L.myEmptyMousePressEvent

        self.view_R = data_view_widget.SliceBox(self.data_R.data_aview.shape[:-1], self.voxel_size, self)
        self.view_R.setCW(self.win_l, 'c')
        self.view_R.setCW(self.win_w, 'w')
        # self.view_R.setSlice(self.data_R.data[0,:,:])
        self.view_R.setSlice(self.data_R.data_aview[...,0])
        if not self.show_view_L:
            self.view_L.setVisible(False)
        if not self.show_view_R:
            self.view_R.setVisible(False)
        # self.view_R.mouseClickSignal.connect(self.add_obj_event)
        # self.view_R.mousePressEvent = self.view_R.myEmptyMousePressEvent
        # self.view_L = data_view_widget.SliceBox(self.data_L.shape[1:])
        # self.update_view_L()
        # self.view_R = data_view_widget.SliceBox(self.data_R.shape[1:])
        # self.update_view_R()

        data_viewer_layout = QtGui.QHBoxLayout()
        # data_viewer_layout.addWidget(self.form_widget)
        data_viewer_layout.addWidget(self.view_L)
        data_viewer_layout.addWidget(self.view_R)
        self.ui.viewer_F.setLayout(data_viewer_layout)

        # connecting callbacks ----------------------------------
        self.ui.view_L_BTN.clicked.connect(self.view_L_callback)
        self.ui.view_R_BTN.clicked.connect(self.view_R_callback)

        # show image data
        self.ui.show_im_L_BTN.clicked.connect(self.show_im_L_callback)
        self.ui.show_im_R_BTN.clicked.connect(self.show_im_R_callback)

        # show label data
        self.ui.show_labels_L_BTN.clicked.connect(self.show_labels_L_callback)
        self.ui.show_labels_R_BTN.clicked.connect(self.show_labels_R_callback)

        # show contours data
        self.ui.show_contours_L_BTN.clicked.connect(self.show_contours_L_callback)
        self.ui.show_contours_R_BTN.clicked.connect(self.show_contours_R_callback)

        # connecting scrollbars
        self.ui.slice_C_SB.valueChanged.connect(self.slider_C_changed)
        self.ui.slice_L_SB.valueChanged.connect(self.slider_L_changed)
        self.ui.slice_R_SB.valueChanged.connect(self.slider_R_changed)

        # to be able to capture key press events immediately
        self.setFocus()

    def keyPressEvent(self, QKeyEvent):
        print 'key event: ',
        key = QKeyEvent.key()
        if key == QtCore.Qt.Key_Escape:
            print 'Escape'
            if self.view_L.area_hist_widget is not None and self.view_L.area_hist_widget.isVisible():
                self.view_L.circle_active = False
                self.view_L.area_hist_widget.close()
                self.view_L.setMouseTracking(False)
                self.view_L.updateSlice()
            else:
                self.close()
        elif key == QtCore.Qt.Key_H:
            print 'H'
            self.action_show_color_model_callback()
        elif key == QtCore.Qt.Key_O:
            print 'O'
            self.action_show_object_list_callback()
        elif key == QtCore.Qt.Key_L:
            print 'R'
            self.run_callback()
        elif key == QtCore.Qt.Key_C:
            print 'C'
            self.action_circle_callback()
        elif key == QtCore.Qt.Key_A:  # interactively add an object
            print 'A'
            self.add_obj_BTN_callback()
        else:
            print key, ' - unrecognized hot key.'

    def add_obj_BTN_callback(self):
        print 'adding object'
        # self.view_L.mouseClickSignal.connect(self.add_obj_event)
        self.view_L.mousePressEvent = self.view_L.myMousePressEvent

    def add_obj_event(self, coords, density):
        print 'add_obj_event - coords: ', coords, ', density: ', density
        center = [self.actual_slice_L, coords[0], coords[1]]
        if self.data_L.objects is not None:
            idx = self.data_L.objects.max() + 1
        else:
            idx = 1
        new_les = Lesion.create_lesion_from_pt(center, density, idx)
        if self.cc.models is not None and self.cc.models['rv_heal'].mean() < density:
            lbl = self.params['hyper_label']
        else:
            lbl = self.params['hypo_label']
        self.data_L.labels[center[0], center[2], center[1]] = lbl
        self.data_L.objects[center[0], center[2], center[1]] = idx

        self.data_L.append_lesion(new_les)

        self.cc.objects_filtration(area=(self.params['min_area'], self.params['max_area']),
                                density=(self.params['min_density'], self.params['max_density']),
                                compactness=self.params['min_compactness'])
        # self.fill_table(self.data_L.lesions, self.data_L.labels)
        # self.fill_table(self.data_L.lesions, self.data_L.labels, self.data_L.labels_filt) #self.cc.filtered_idxs)
        self.fill_table(self.cc.actual_data.lesions, self.cc.actual_data.labels, self.cc.filtered_idxs)
        # self.actual_data.labels_filt
        self.ui.show_contours_L_BTN.setEnabled(True)
        #TODO: aktualizovat tabulku

    def min_compactness_SL_changed(self, value):
        self.objects_widget.ui.min_compactness_LE.setText('%.3f' % (value * self.params['compactness_step']))
        self.object_slider_changed(value)

    def min_compactness_LE_changed(self, value):
        try:  # must be due to the possibility that no character could be entered
            self.objects_widget.ui.min_compactness_SL.setValue(int(value * self.objects_widget.ui.min_compactness_SL.maximum()))
            # self.params['compactness'] = value
        except:
            pass

    def action_circle_callback(self):
        self.view_L.circle_active = True
        self.view_L.setMouseTracking(True)
        self.view_L.area_hist_widget = ahw.AreaHistWidget()
        self.view_L.area_hist_widget.show()
        # self.view_R.circle_active = True

    def action_show_color_model_callback(self):
        if self.hist_widget is None:
            self.hist_widget = Hist_widget(data=self.data_L.data)
        self.hist_widget.show()
        self.hist_widget.setFocus()

    def action_show_object_list_callback(self):
        # if self.objects_widget is None:
        #     self.objects_widget = Objects_widget()
        #     self.objects_widget.area_RS.endValueChanged.connect(self.max_area_changed_callback)
        #     self.objects_widget.area_RS.startValueChanged.connect(self.min_area_changed_callback)
        self.objects_widget.show()
        self.objects_widget.setFocus()

    def serie_1_RB_callback(self):
        self.cc.active_serie = 1
        self.cc.actual_data = self.cc.data_1

    def serie_2_RB_callback(self):
        self.cc.active_serie = 2
        self.cc.actual_data = self.cc.data_2

    def selection_changed(self, selected, deselected):
        #TODO: povolit oznaceni pouze jednoho objektu?
        if selected.indexes():
            self.selected_objects_labels = [self.table_model.objects[x.row()].label for x in self.objects_widget.ui.objects_TV.selectionModel().selectedRows()]
            # print 'show only', self.selected_objects_labels
            slice = [int(x.center[0]) for x in self.cc.actual_data.lesions if x.label == self.selected_objects_labels[0]][0]
            self.ui.slice_C_SB.setValue(slice)
        else:
            self.selected_objects_labels = [x.label for x in self.table_model.objects]
            # print 'show all', self.selected_objects_labels
        min_area, max_area = self.objects_widget.area_RS.getRange()
        min_density, max_density = self.objects_widget.density_RS.getRange()
        min_comp = self.objects_widget.ui.min_compactness_SL.value() / self.params['compactness_step']
        self.cc.objects_filtration(self.selected_objects_labels, area=(min_area, max_area), density=(min_density, max_density),
                                   compactness=min_comp)
        #TODO: nasleduje prasarna
        if self.view_L.show_mode == self.view_L.SHOW_LABELS:
            self.show_labels_L_callback()
        if self.view_R.show_mode == self.view_R.SHOW_LABELS:
            self.show_labels_R_callback()

    def object_slider_changed(self, value):
        min_area = self.objects_widget.area_RS.start()
        max_area = self.objects_widget.area_RS.end()
        min_density = self.objects_widget.density_RS.start()
        max_density = self.objects_widget.density_RS.end()
        min_comp = self.objects_widget.ui.min_compactness_SL.value() * self.params['compactness_step']
        self.params['max_area'] = max_area
        self.params['min_area'] = min_area
        self.params['max_density'] = max_density
        self.params['min_density'] = min_density
        self.params['min_compactness'] = min_comp

        self.selected_objects_labels

        self.cc.objects_filtration(#self.selected_objects_labels,
                                   area= (min_area, max_area),
                                   density=(min_density, max_density),
                                   compactness=min_comp)
        # self.fill_table(self.cc.labels[self.cc.filtered_idxs], self.cc.areas[self.cc.filtered_idxs], self.cc.comps[self.cc.filtered_idxs])
        if self.cc.filtered_idxs is not None:
            self.fill_table(self.cc.actual_data.lesions, self.cc.actual_data.labels, self.cc.filtered_idxs)
            # TODO: nasleduje prasarna
            if self.view_L.show_mode == self.view_L.SHOW_LABELS:
                self.show_labels_L_callback()
            if self.view_R.show_mode == self.view_R.SHOW_LABELS:
                self.show_labels_R_callback()

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

    def scroll_event(self, value, who):
        if who == 0:  # left viewer
            new = self.actual_slice_L + value
            if (new < 0) or (new >= self.data_L.n_slices):
                return
        elif who == 1:  # right viewer
            new = self.actual_slice_R + value
            if (new < 0) or (new >= self.data_R.n_slices):
                return

    def slider_C_changed(self, val):
        if val == self.actual_slice_L:
            return

        if (val >= 0) and (val < self.data_L.n_slices):
            diff = val - self.actual_slice_L
            self.actual_slice_L = val
        else:
            return

        new_slice_R = self.actual_slice_R + diff
        if new_slice_R < 0:
            new_slice_R = 0
        elif new_slice_R >= self.data_R.n_slices:
            new_slice_R = self.data_R.n_slices - 1

        self.actual_slice_R = new_slice_R

        self.ui.slice_L_SB.setValue(self.actual_slice_L)
        self.ui.slice_R_SB.setValue(self.actual_slice_R)

        im_L = self.get_image('L')
        im_R = self.get_image('R')
        if self.show_mode_L == SHOW_CONTOURS:
            labels_L = self.data_L.labels_filt[self.actual_slice_L, :, :]
            obj_centers_L = self.data_L.object_centers[self.actual_slice_L, ...]
        else:
            labels_L = None
            obj_centers_L = None
        if self.show_mode_R == SHOW_CONTOURS:
            labels_R = self.data_R.labels_filt[self.actual_slice_R, :, :]
            obj_centers_R = self.data_R.object_centers[self.actual_slice_R, ...]
        else:
            labels_R = None
            obj_centers_R = None

        self.view_L.setSlice(im_L, contours=labels_L, centers=obj_centers_L)
        self.view_R.setSlice(im_R, contours=labels_R, centers=obj_centers_R)

        self.ui.slice_number_L_LBL.setText('%i/%i' % (self.actual_slice_L + 1, self.data_L.n_slices))
        self.ui.slice_number_C_LBL.setText('slice # = %i/%i' % (self.actual_slice_L + 1, self.data_L.n_slices))

    def slider_L_changed(self, val):
        if val == self.actual_slice_L:
            return

        if (val >= 0) and (val < self.data_L.n_slices):
            self.actual_slice_L = val
        else:
            return

        self.ui.slice_C_SB.setValue(self.actual_slice_L)

        im_L = self.get_image('L')
        if self.show_mode_L == SHOW_CONTOURS:
            labels_L = self.data_L.labels_filt[self.actual_slice_L, :, :]
            obj_centers = self.data_L.object_centers[self.actual_slice_L, ...]
        else:
            labels_L = None
            obj_centers = None

        self.view_L.setSlice(im_L, contours=labels_L, centers=obj_centers)

        self.ui.slice_number_L_LBL.setText('%i/%i' % (self.actual_slice_L + 1, self.data_L.n_slices))
        self.ui.slice_number_C_LBL.setText('slice # = %i/%i' % (self.actual_slice_L + 1, self.data_L.n_slices))

    def slider_R_changed(self, val):
        if (val >= 0) and (val < self.data_R.n_slices):
            self.actual_slice_R = val
        else:
            return

        self.ui.slice_number_R_LBL.setText('%i/%i' % (self.actual_slice_R + 1, self.data_R.n_slices))

        im_R = self.get_image('R')
        if self.show_mode_R == SHOW_CONTOURS:
            labels_R = self.data_R.labels_filt[self.actual_slice_R, :, :]
            obj_centers = self.data_R.object_centers[self.actual_slice_R, ...]
        else:
            labels_R = None
            obj_centers = None

        self.view_R.setSlice(im_R, contours=labels_R, centers=obj_centers)

    def calculate_models_callback(self):
        self.statusBar().showMessage('Calculating intensity models...')
        if self.params['zoom']:
            data = self.data_zoom(self.cc.actual_data.data, self.cc.actual_data.voxel_size, self.params['working_voxel_size_mm'])
            mask = self.data_zoom(self.cc.actual_data.mask, self.cc.actual_data.voxel_size, self.params['working_voxel_size_mm'])
        else:
            data = tools.resize3D(self.cc.actual_data.data, self.params['scale'], sliceId=0)
            mask = tools.resize3D(self.cc.actual_data.mask, self.params['scale'], sliceId=0)
        self.cc.models = self.cc.calculate_intensity_models(data, mask)
        self.statusBar().showMessage('Intensity models calculated.')
        self.update_models()

    def update_models(self):
        if self.hist_widget is None:
            self.hist_widget = Hist_widget(data=self.data_L.data)
        self.hist_widget.update_heal_rv(self.cc.models['rv_heal'])
        self.hist_widget.update_hypo_rv(self.cc.models['rv_hypo'])
        self.hist_widget.update_hyper_rv(self.cc.models['rv_hyper'])

        self.hist_widget.update_figures()

    def run_callback(self):
        # Viewer_3D.run(self.data)
        # viewer = Viewer_3D.Viewer_3D(self.data)
        # viewer.show()

        # run localization
        self.statusBar().showMessage('Localization started...')
        self.cc.run()
        self.update_models()

        # filling table with objects
        # self.fill_table(self.cc.labels[self.cc.filtered_idxs], self.cc.areas[self.cc.filtered_idxs], self.cc.comps[self.cc.filtered_idxs])
        self.fill_table(self.cc.actual_data.lesions, self.cc.actual_data.objects, self.cc.filtered_idxs)
        self.selected_objects_labels = [x.label for x in self.table_model.objects]
        # seting up range of area slider
        areas = [x.area for x in self.cc.actual_data.lesions]
        self.objects_widget.set_area_range(areas)

        self.ui.show_labels_L_BTN.setEnabled(True)
        self.ui.show_contours_L_BTN.setEnabled(True)

        # self.cc.areas = np.array([10, 20, 30, 8])
        # self.cc.comps = np.array([51, 60, 70, 80])
        # self.cc.objects = np.array([[0,0,-1,3,3],[0,-1,-1,3,3],[-1,-1,-1,-1,-1],[1,1,-1,-1,2],[1,-1,-1,2,2]])
        # self.cc.labels = np.unique(self.cc.objects)[1:]
        # self.cc.n_objects = len(np.unique(self.cc.objects)) - 1
        # self.fill_table(self.cc.labels, self.cc.areas, self.cc.comps)

    def fill_table(self, lesions, labels, idxs=None):
        # if self.objects_widget is None:
        #     self.objects_widget = Objects_widget()
        #     # tableview selection changed
        #     self.objects_widget.ui.objects_TV.selectionModel().selectionChanged.connect(self.selection_changed)
        if idxs is None:
            idxs = [x.label for x in lesions]
            # lesions_filtered = lesions[:]  # copying the list
        # else:
        lesions_filtered = [x for x in lesions if x.label in idxs]
        # labels_filtered = np.where(labels in idxs, labels, 0)
        labels_filtered = labels * np.in1d(labels, idxs).reshape(labels.shape)
        self.table_model = mtm.MyTableModel(lesions_filtered, labels_filtered)
        self.objects_widget.ui.objects_TV.setModel(self.table_model)
        self.objects_widget.ui.objects_TV.selectionModel().selectionChanged.connect(self.selection_changed)

        # self.ui.objects_TV.selectionModel().selectionChanged.connect(self.selection_changed)

    # def fill_table(self, labels, areas, comps):
    #     features = np.vstack((labels, areas, comps)).T
    #     self.table_model = mtm.MyTableModel(features)
    #     self.ui.objects_TV.setModel(self.table_model)
    #     self.ui.objects_TV.selectionModel().selectionChanged.connect(self.selection_changed)
    #     # self.tableview.verticalHeader().setVisible(True)


    def view_L_callback(self):
        self.show_view_L = not self.show_view_L
        self.view_L.setVisible(self.show_view_L)

        # enabling and disabling other toolbar icons
        self.ui.show_im_L_BTN.setEnabled(self.show_view_L)

        if self.show_view_L and self.data_L.labels is not None:
            self.ui.show_labels_L_BTN.setEnabled(True)
            self.ui.show_contours_L_BTN.setEnabled(True)
        else:
            self.ui.show_labels_L_BTN.setEnabled(False)
            self.ui.show_contours_L_BTN.setEnabled(False)

        # if not self.show_view_L:
            # self.ui.viewer_F.setFixedWidth(self.ui.viewer_F.width()/2)
            # self.resize(self.height(), self.width() - self.ui.viewer_F.width()/2)

        self.statusBar().showMessage('Left view set to %s' % self.show_view_L)
        # print 'view_1 set to', self.show_view_1

        # self.form_widget.update_figures()
        self.view_L.update()

    def view_R_callback(self):
        self.show_view_R = not self.show_view_R
        self.view_R.setVisible(self.show_view_R)

        # enabling and disabling other toolbar icons
        self.ui.show_im_R_BTN.setEnabled(not self.ui.show_im_R_BTN.isEnabled())

        if self.show_view_R and self.data_R.labels is not None:
            self.ui.show_labels_R_BTN.setEnabled(True)
            self.ui.show_contours_R_BTN.setEnabled(True)
        else:
            self.ui.show_labels_R_BTN.setEnabled(False)
            self.ui.show_contours_R_BTN.setEnabled(False)

        # if not self.show_view_R:
        #     self.ui.viewer_F.setFixedWidth(self.ui.viewer_F.width()/2)

        self.statusBar().showMessage('Right view set to %s' % self.show_view_R)
        # print 'view_2 set to', self.show_view_2

        # self.form_widget.update_figures()
        self.view_R.update()

    def show_im_L_callback(self):
        self.show_mode_L = SHOW_IM
        self.view_L.show_mode = self.view_L.SHOW_IM

        im = self.get_image('L')
        self.view_L.setSlice(im)

        self.statusBar().showMessage('data_L set to im')

    def show_im_R_callback(self):
        self.show_mode_R = SHOW_IM
        self.view_R.show_mode = self.view_R.SHOW_IM

        im = self.get_image('R')
        self.view_R.setSlice(im)

        self.statusBar().showMessage('data_R set to im')

    def show_labels_L_callback(self):
        self.show_mode_L = SHOW_LABELS
        self.view_L.show_mode = self.view_L.SHOW_LABELS

        im = self.get_image('L')
        print im.dtype
        self.view_L.setSlice(im)

        self.statusBar().showMessage('data_L set to labels')

    def show_labels_R_callback(self):
        self.show_mode_R = SHOW_LABELS
        self.view_R.show_mode = self.view_R.SHOW_LABELS

        im = self.get_image('R')
        self.view_R.setSlice(im)

        self.statusBar().showMessage('data_R set to labels')

    def show_contours_L_callback(self):
        self.show_mode_L = SHOW_CONTOURS
        self.view_L.show_mode = self.view_L.SHOW_CONTOURS

        im = self.get_image('L')
        labels = self.data_L.labels_filt[self.actual_slice_L, :, :]
        # obj_centers = [self.data_L.object_centers[:2] if self.data_L.object_centers[2] == self.actual_slice_L]
        # obj_centers = [x[1:] for x in self.data_L.object_centers if round(x[0]) == self.actual_slice_L]
        self.view_L.setSlice(im, contours=labels, centers=self.data_L.object_centers[self.actual_slice_L,...])

        self.statusBar().showMessage('data_L set to contours')

    def show_contours_R_callback(self):
        self.show_mode_R = SHOW_CONTOURS
        self.view_R.show_mode = self.view_R.SHOW_CONTOURS

        im = self.get_image('R')
        labels = self.data_R.labels_filt[self.actual_slice_R, :, :]
        self.view_R.setSlice(im, contours=labels)

        self.statusBar().showMessage('data_R set to contours')

    def figure_L_CB_callback(self):
        if self.ui.figure_L_CB.currentIndex() == 0:
            self.data_L = self.cc.data_1
        elif self.ui.figure_L_CB.currentIndex() == 1:
            self.data_L = self.cc.data_2

        if self.actual_slice_L >= self.data_L.n_slices:
            self.actual_slice_L = self.data_L.n_slices - 1

        self.ui.slice_L_SB.setMaximum(self.data_L.n_slices - 1)
        self.ui.slice_C_SB.setMaximum(self.data_L.n_slices - 1)

        if (self.data_L.labels is not None) and self.show_view_L:
            self.ui.show_labels_L_BTN.setEnabled(True)
            self.ui.show_contours_L_BTN.setEnabled(True)
        else:
            self.ui.show_labels_L_BTN.setEnabled(False)
            self.ui.show_contours_L_BTN.setEnabled(False)

        self.view_L.reinit((self.data_L.shape[2], self.data_L.shape[1]))
        self.show_im_L_callback()

    def figure_R_CB_callback(self):
        if self.ui.figure_R_CB.currentIndex() == 0:
            self.data_R = self.cc.data_1
        elif self.ui.figure_R_CB.currentIndex() == 1:
            self.data_R = self.cc.data_2

        if self.actual_slice_R >= self.data_R.n_slices:
            self.actual_slice_R = self.data_R.n_slices - 1

        self.ui.slice_R_SB.setMaximum(self.data_R.n_slices - 1)

        if (self.data_R.labels is not None) and self.show_view_R:
            self.ui.show_labels_R_BTN.setEnabled(True)
            self.ui.show_contours_R_BTN.setEnabled(True)
        else:
            self.ui.show_labels_R_BTN.setEnabled(False)
            self.ui.show_contours_R_BTN.setEnabled(False)

        # self.view_R.reinit(self.data_R.shape[2:0:-1])
        self.view_R.reinit((self.data_R.shape[2], self.data_R.shape[1]))
        self.show_im_R_callback()

    def action_load_serie_callback(self, serie_number):
        fname = QtGui.QFileDialog.getOpenFileName(self, 'Open file', self.params['data_dir'])
        print 'Does not work yet.'
        print fname

    def remove_obj_BTN_callback(self):
        indexes = self.ui.objects_TV.selectionModel().selectedRows()
        # for i in reversed(indexes):
        #     self.table_model.removeRow(i.row())
        for i in indexes:
            obj = self.table_model.objects[i.row()]
            self.remove_object(obj)

        # self.cc.actual_data.filtered_idxs = [x.label for x in self.cc.actual_data.lesions]
        self.selected_objects_labels = [x.label for x in self.cc.actual_data.lesions]
        self.cc.objects_filtration(self.selected_objects_labels, min_area=self.ui.min_area_SL.value(), max_area=self.ui.max_area_SL.value())
        self.fill_table(self.cc.actual_data.lesions, self.cc.actual_data.objects, self.cc.filtered_idxs)
        #TODO: nasleduje prasarna
        if self.view_L.show_mode == self.view_L.SHOW_LABELS:
            self.show_labels_L_callback()
        if self.view_R.show_mode == self.view_R.SHOW_LABELS:
            self.show_labels_R_callback()

    def remove_object(self, obj):
        lbl = obj.label
        im = self.cc.actual_data.objects == lbl
        self.cc.actual_data.objects[np.nonzero(im)] = self.params['bgd_label']
        self.cc.actual_data.labels[np.nonzero(im)] = self.params['healthy_label']

        self.cc.actual_data.lesions.remove(obj)
        print 'removed label', lbl

    def get_image(self, site):
        im = None
        if site == 'L':
            if self.show_mode_L == SHOW_IM or self.show_mode_L == SHOW_CONTOURS:
                im = self.data_L.data_aview[..., self.actual_slice_L]
            elif self.show_mode_L == SHOW_LABELS:
                # im = self.data_L.labels_aview[...,self.actual_slice_L]
                im = self.data_L.labels_filt_aview[..., self.actual_slice_L]
            # elif self.show_mode_L == SHOW_FILTERED_LABELS:
            #     im = self.data_L.labels_filt_aview[...,self.actual_slice_L]
        elif site == 'R':
            if self.show_mode_R == SHOW_IM or self.show_mode_R == SHOW_CONTOURS:
                im = self.data_R.data_aview[..., self.actual_slice_R]
            elif self.show_mode_R == SHOW_LABELS:
                # im = self.data_R.labels_aview[...,self.actual_slice_R]
                im = self.data_R.labels_filt_aview[..., self.actual_slice_R]
            # elif self.show_mode_R == SHOW_FILTERED_LABELS:
            #     im = self.data_R.labels_filt_aview[...,self.actual_slice_R]
        return im

    def run(self, im, labels, healthy_label, hypo_label, hyper_label, slice_axis=2, disp_smoothed=False):
        if slice_axis == 0:
            im = np.transpose(im, (1, 2, 0))
            labels = np.transpose(labels, (1, 2, 0))
        app = QtGui.QApplication(sys.argv)
        le = LessionEditor(im, labels, healthy_label, hypo_label, hyper_label, disp_smoothed)
        le.show()
        sys.exit(app.exec_())

################################################################################
################################################################################
if __name__ == '__main__':
    fname = ''
    fnames = list()

    # 2 hypo, 1 on the border --------------------
    # arterial 0.6mm - bad
    # fname = '/home/tomas/Data/liver_segmentation_06mm/tryba/data_other/org-exp_183_46324212_arterial_0.6_B30f-.pklz'
    # venous 0.6mm - good
    # fname = '/home/tomas/Data/liver_segmentation_06mm/tryba/data_other/org-exp_183_46324212_venous_0.6_B20f-.pklz'
    # venous 5mm - ok, but wrong approach
    # TODO: study ID 29 - 3/3, 2 velke spoji v jeden
    # slice = 17
    fnames.append('/home/tomas/Data/liver_segmentation/tryba/data_other/org-exp_183_46324212_venous_5.0_B30f-.pklz')
    fnames.append('/home/tomas/Data/liver_segmentation/tryba/data_other/org-exp_183_46324212_arterial_5.0_B30f-.pklz')

    # hypo in venous -----------------------
    # arterial - bad
    # fname = '/home/tomas/Data/liver_segmentation_06mm/tryba/data_other/org-exp_186_49290986_venous_0.6_B20f-.pklz'
    # venous - good
    # fname = '/home/tomas/Data/liver_segmentation_06mm/tryba/data_other/org-exp_186_49290986_arterial_0.6_B20f-.pklz'
    # fnames.append('/home/tomas/Data/liver_segmentation/tryba/data_other/org-exp_186_49290986_venous_0.6_B20f-.pklz')
    # fnames.append('/home/tomas/Data/liver_segmentation/tryba/data_other/org-exp_186_49290986_arterial_0.6_B20f-.pklz')

    # hyper, 1 on the border -------------------
    # arterial 0.6mm - not that bad
    # fname = '/home/tomas/Data/liver_segmentation_06mm/hyperdenzni/org-exp_239_61293268_DE_Art_Abd_0.75_I26f_M_0.5-.pklz'
    # venous 5mm - bad
    # fname = '/home/tomas/Data/liver_segmentation_06mm/hyperdenzni/org-exp_239_61293268_DE_Ven_Abd_0.75_I26f_M_0.5-.pklz'

    # shluk -----------------
    # arterial 5mm
    # fname = '/home/tomas/Data/liver_segmentation/tryba/data_other/org-exp_180_49509315_arterial_5.0_B30f-.pklz'
    # fname = '/home/tomas/Data/liver_segmentation_06mm/tryba/data_other/org-exp_180_49509315_arterial_0.6_B20f-.pklz'
    # TODO: study ID 18 - velke najde (hyper v arterialni fazi), 1/2 z malych
    # fnames.append('/home/tomas/Data/liver_segmentation/tryba/data_other/org-exp_180_49509315_venous_5.0_B30f-.pklz')
    # fnames.append('/home/tomas/Data/liver_segmentation/tryba/data_other/org-exp_180_49509315_arterial_5.0_B30f-.pklz')

    # targeted
    # arterial 0.6mm - bad
    # fname = '/home/tomas/Data/liver_segmentation_06mm/hyperdenzni/org-exp_238_54280551_Abd_Arterial_0.75_I26f_3-.pklz'
    # venous 0.6mm - bad
    # fname = '/home/tomas/Data/liver_segmentation_06mm/hyperdenzni/org-exp_238_54280551_Abd_Venous_0.75_I26f_3-.pklz'

    # TODO: study ID 25 - 2/2
    # fnames.append('/home/tomas/Data/liver_segmentation/tryba/data_other/org-exp_185_48441644_venous_5.0_B30f-.pklz')
    # fnames.append('/home/tomas/Data/liver_segmentation/tryba/data_other/org-exp_185_48441644_arterial_5.0_B30f-.pklz')

    # TODO: study ID 21 - 0/2, nenasel ani jeden pekny hypo
    # slice = 6
    # fnames.append('/home/tomas/Data/liver_segmentation/tryba/data_other/org-exp_186_49290986_venous_5.0_B30f-.pklz')
    # fnames.append('/home/tomas/Data/liver_segmentation/tryba/data_other/org-exp_186_49290986_arterial_5.0_B30f-.pklz')

    # runing application -------------------------
    app = QtGui.QApplication(sys.argv)
    # app = MyApplication(sys.argv)
    le = LessionEditor(fnames)
    le.show()
    sys.exit(app.exec_())