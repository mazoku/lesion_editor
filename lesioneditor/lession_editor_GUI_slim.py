# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'lession_editor_GUI_slim.ui'
#
# Created: Tue Sep  8 15:27:16 2015
#      by: PyQt4 UI code generator 4.10.4
#
# WARNING! All changes made in this file will be lost!

from PyQt4 import QtCore, QtGui

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName(_fromUtf8("MainWindow"))
        MainWindow.resize(721, 839)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        MainWindow.setTabShape(QtGui.QTabWidget.Rounded)
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.verticalLayout_4 = QtGui.QVBoxLayout(self.centralwidget)
        self.verticalLayout_4.setObjectName(_fromUtf8("verticalLayout_4"))
        self.splitter = QtGui.QSplitter(self.centralwidget)
        self.splitter.setOrientation(QtCore.Qt.Horizontal)
        self.splitter.setObjectName(_fromUtf8("splitter"))
        self.frame_R = QtGui.QFrame(self.splitter)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_R.sizePolicy().hasHeightForWidth())
        self.frame_R.setSizePolicy(sizePolicy)
        self.frame_R.setFrameShape(QtGui.QFrame.StyledPanel)
        self.frame_R.setFrameShadow(QtGui.QFrame.Raised)
        self.frame_R.setObjectName(_fromUtf8("frame_R"))
        self.verticalLayout_2 = QtGui.QVBoxLayout(self.frame_R)
        self.verticalLayout_2.setObjectName(_fromUtf8("verticalLayout_2"))
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.verticalLayout_7 = QtGui.QVBoxLayout()
        self.verticalLayout_7.setObjectName(_fromUtf8("verticalLayout_7"))
        self.figure_L_CB = QtGui.QComboBox(self.frame_R)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.figure_L_CB.sizePolicy().hasHeightForWidth())
        self.figure_L_CB.setSizePolicy(sizePolicy)
        self.figure_L_CB.setObjectName(_fromUtf8("figure_L_CB"))
        self.verticalLayout_7.addWidget(self.figure_L_CB)
        self.left_view_btns_F = QtGui.QFrame(self.frame_R)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Maximum, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.left_view_btns_F.sizePolicy().hasHeightForWidth())
        self.left_view_btns_F.setSizePolicy(sizePolicy)
        self.left_view_btns_F.setMinimumSize(QtCore.QSize(0, 20))
        self.left_view_btns_F.setFrameShape(QtGui.QFrame.StyledPanel)
        self.left_view_btns_F.setFrameShadow(QtGui.QFrame.Raised)
        self.left_view_btns_F.setObjectName(_fromUtf8("left_view_btns_F"))
        self.horizontalLayout_12 = QtGui.QHBoxLayout(self.left_view_btns_F)
        self.horizontalLayout_12.setObjectName(_fromUtf8("horizontalLayout_12"))
        self.view_L_BTN = QtGui.QPushButton(self.left_view_btns_F)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.view_L_BTN.sizePolicy().hasHeightForWidth())
        self.view_L_BTN.setSizePolicy(sizePolicy)
        self.view_L_BTN.setMinimumSize(QtCore.QSize(40, 0))
        self.view_L_BTN.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.view_L_BTN.setText(_fromUtf8(""))
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(_fromUtf8("icons/Eye.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.view_L_BTN.setIcon(icon)
        self.view_L_BTN.setIconSize(QtCore.QSize(24, 24))
        self.view_L_BTN.setObjectName(_fromUtf8("view_L_BTN"))
        self.horizontalLayout_12.addWidget(self.view_L_BTN)
        self.show_im_L_BTN = QtGui.QPushButton(self.left_view_btns_F)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.show_im_L_BTN.sizePolicy().hasHeightForWidth())
        self.show_im_L_BTN.setSizePolicy(sizePolicy)
        self.show_im_L_BTN.setMinimumSize(QtCore.QSize(40, 0))
        self.show_im_L_BTN.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.show_im_L_BTN.setText(_fromUtf8(""))
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(_fromUtf8("icons/Stock graph.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.show_im_L_BTN.setIcon(icon1)
        self.show_im_L_BTN.setIconSize(QtCore.QSize(24, 24))
        self.show_im_L_BTN.setObjectName(_fromUtf8("show_im_L_BTN"))
        self.horizontalLayout_12.addWidget(self.show_im_L_BTN)
        self.show_labels_L_BTN = QtGui.QPushButton(self.left_view_btns_F)
        self.show_labels_L_BTN.setEnabled(False)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.show_labels_L_BTN.sizePolicy().hasHeightForWidth())
        self.show_labels_L_BTN.setSizePolicy(sizePolicy)
        self.show_labels_L_BTN.setMinimumSize(QtCore.QSize(40, 0))
        self.show_labels_L_BTN.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.show_labels_L_BTN.setText(_fromUtf8(""))
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap(_fromUtf8("icons/Blue tag.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.show_labels_L_BTN.setIcon(icon2)
        self.show_labels_L_BTN.setIconSize(QtCore.QSize(24, 24))
        self.show_labels_L_BTN.setObjectName(_fromUtf8("show_labels_L_BTN"))
        self.horizontalLayout_12.addWidget(self.show_labels_L_BTN)
        self.show_contours_L_BTN = QtGui.QPushButton(self.left_view_btns_F)
        self.show_contours_L_BTN.setEnabled(False)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.show_contours_L_BTN.sizePolicy().hasHeightForWidth())
        self.show_contours_L_BTN.setSizePolicy(sizePolicy)
        self.show_contours_L_BTN.setMinimumSize(QtCore.QSize(40, 0))
        self.show_contours_L_BTN.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.show_contours_L_BTN.setText(_fromUtf8(""))
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap(_fromUtf8("icons/Brush.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.show_contours_L_BTN.setIcon(icon3)
        self.show_contours_L_BTN.setIconSize(QtCore.QSize(24, 24))
        self.show_contours_L_BTN.setObjectName(_fromUtf8("show_contours_L_BTN"))
        self.horizontalLayout_12.addWidget(self.show_contours_L_BTN)
        self.verticalLayout_7.addWidget(self.left_view_btns_F)
        self.horizontalLayout.addLayout(self.verticalLayout_7)
        self.line_4 = QtGui.QFrame(self.frame_R)
        self.line_4.setFrameShape(QtGui.QFrame.VLine)
        self.line_4.setFrameShadow(QtGui.QFrame.Sunken)
        self.line_4.setObjectName(_fromUtf8("line_4"))
        self.horizontalLayout.addWidget(self.line_4)
        self.verticalLayout_8 = QtGui.QVBoxLayout()
        self.verticalLayout_8.setObjectName(_fromUtf8("verticalLayout_8"))
        self.figure_R_CB = QtGui.QComboBox(self.frame_R)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.figure_R_CB.sizePolicy().hasHeightForWidth())
        self.figure_R_CB.setSizePolicy(sizePolicy)
        self.figure_R_CB.setObjectName(_fromUtf8("figure_R_CB"))
        self.verticalLayout_8.addWidget(self.figure_R_CB)
        self.right_view_btns_F = QtGui.QFrame(self.frame_R)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Maximum, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.right_view_btns_F.sizePolicy().hasHeightForWidth())
        self.right_view_btns_F.setSizePolicy(sizePolicy)
        self.right_view_btns_F.setMinimumSize(QtCore.QSize(0, 20))
        self.right_view_btns_F.setFrameShape(QtGui.QFrame.StyledPanel)
        self.right_view_btns_F.setFrameShadow(QtGui.QFrame.Raised)
        self.right_view_btns_F.setObjectName(_fromUtf8("right_view_btns_F"))
        self.horizontalLayout_13 = QtGui.QHBoxLayout(self.right_view_btns_F)
        self.horizontalLayout_13.setObjectName(_fromUtf8("horizontalLayout_13"))
        self.view_R_BTN = QtGui.QPushButton(self.right_view_btns_F)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.view_R_BTN.sizePolicy().hasHeightForWidth())
        self.view_R_BTN.setSizePolicy(sizePolicy)
        self.view_R_BTN.setMinimumSize(QtCore.QSize(40, 0))
        self.view_R_BTN.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.view_R_BTN.setText(_fromUtf8(""))
        self.view_R_BTN.setIcon(icon)
        self.view_R_BTN.setIconSize(QtCore.QSize(24, 24))
        self.view_R_BTN.setObjectName(_fromUtf8("view_R_BTN"))
        self.horizontalLayout_13.addWidget(self.view_R_BTN)
        self.show_im_R_BTN = QtGui.QPushButton(self.right_view_btns_F)
        self.show_im_R_BTN.setEnabled(False)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.show_im_R_BTN.sizePolicy().hasHeightForWidth())
        self.show_im_R_BTN.setSizePolicy(sizePolicy)
        self.show_im_R_BTN.setMinimumSize(QtCore.QSize(40, 0))
        self.show_im_R_BTN.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.show_im_R_BTN.setText(_fromUtf8(""))
        self.show_im_R_BTN.setIcon(icon1)
        self.show_im_R_BTN.setIconSize(QtCore.QSize(24, 24))
        self.show_im_R_BTN.setObjectName(_fromUtf8("show_im_R_BTN"))
        self.horizontalLayout_13.addWidget(self.show_im_R_BTN)
        self.show_labels_R_BTN = QtGui.QPushButton(self.right_view_btns_F)
        self.show_labels_R_BTN.setEnabled(False)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.show_labels_R_BTN.sizePolicy().hasHeightForWidth())
        self.show_labels_R_BTN.setSizePolicy(sizePolicy)
        self.show_labels_R_BTN.setMinimumSize(QtCore.QSize(40, 0))
        self.show_labels_R_BTN.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.show_labels_R_BTN.setText(_fromUtf8(""))
        self.show_labels_R_BTN.setIcon(icon2)
        self.show_labels_R_BTN.setIconSize(QtCore.QSize(24, 24))
        self.show_labels_R_BTN.setObjectName(_fromUtf8("show_labels_R_BTN"))
        self.horizontalLayout_13.addWidget(self.show_labels_R_BTN)
        self.show_contours_R_BTN = QtGui.QPushButton(self.right_view_btns_F)
        self.show_contours_R_BTN.setEnabled(False)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.show_contours_R_BTN.sizePolicy().hasHeightForWidth())
        self.show_contours_R_BTN.setSizePolicy(sizePolicy)
        self.show_contours_R_BTN.setMinimumSize(QtCore.QSize(40, 0))
        self.show_contours_R_BTN.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.show_contours_R_BTN.setText(_fromUtf8(""))
        self.show_contours_R_BTN.setIcon(icon3)
        self.show_contours_R_BTN.setIconSize(QtCore.QSize(24, 24))
        self.show_contours_R_BTN.setObjectName(_fromUtf8("show_contours_R_BTN"))
        self.horizontalLayout_13.addWidget(self.show_contours_R_BTN)
        self.verticalLayout_8.addWidget(self.right_view_btns_F)
        self.horizontalLayout.addLayout(self.verticalLayout_8)
        self.verticalLayout_2.addLayout(self.horizontalLayout)
        self.viewer_F = QtGui.QFrame(self.frame_R)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.viewer_F.sizePolicy().hasHeightForWidth())
        self.viewer_F.setSizePolicy(sizePolicy)
        self.viewer_F.setMinimumSize(QtCore.QSize(0, 0))
        self.viewer_F.setFrameShape(QtGui.QFrame.Box)
        self.viewer_F.setObjectName(_fromUtf8("viewer_F"))
        self.verticalLayout_2.addWidget(self.viewer_F)
        self.frame_7 = QtGui.QFrame(self.frame_R)
        self.frame_7.setFrameShape(QtGui.QFrame.StyledPanel)
        self.frame_7.setFrameShadow(QtGui.QFrame.Raised)
        self.frame_7.setObjectName(_fromUtf8("frame_7"))
        self.horizontalLayout_3 = QtGui.QHBoxLayout(self.frame_7)
        self.horizontalLayout_3.setObjectName(_fromUtf8("horizontalLayout_3"))
        self.slice_number_L_LBL = QtGui.QLabel(self.frame_7)
        self.slice_number_L_LBL.setMaximumSize(QtCore.QSize(60, 16777215))
        self.slice_number_L_LBL.setObjectName(_fromUtf8("slice_number_L_LBL"))
        self.horizontalLayout_3.addWidget(self.slice_number_L_LBL)
        self.slice_L_SB = QtGui.QScrollBar(self.frame_7)
        self.slice_L_SB.setOrientation(QtCore.Qt.Horizontal)
        self.slice_L_SB.setObjectName(_fromUtf8("slice_L_SB"))
        self.horizontalLayout_3.addWidget(self.slice_L_SB)
        self.slice_R_SB = QtGui.QScrollBar(self.frame_7)
        self.slice_R_SB.setEnabled(True)
        self.slice_R_SB.setOrientation(QtCore.Qt.Horizontal)
        self.slice_R_SB.setObjectName(_fromUtf8("slice_R_SB"))
        self.horizontalLayout_3.addWidget(self.slice_R_SB)
        self.slice_number_R_LBL = QtGui.QLabel(self.frame_7)
        self.slice_number_R_LBL.setMaximumSize(QtCore.QSize(60, 16777215))
        self.slice_number_R_LBL.setObjectName(_fromUtf8("slice_number_R_LBL"))
        self.horizontalLayout_3.addWidget(self.slice_number_R_LBL)
        self.verticalLayout_2.addWidget(self.frame_7)
        self.horizontalLayout_11 = QtGui.QHBoxLayout()
        self.horizontalLayout_11.setObjectName(_fromUtf8("horizontalLayout_11"))
        self.slice_number_C_LBL = QtGui.QLabel(self.frame_R)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.slice_number_C_LBL.sizePolicy().hasHeightForWidth())
        self.slice_number_C_LBL.setSizePolicy(sizePolicy)
        self.slice_number_C_LBL.setMinimumSize(QtCore.QSize(100, 0))
        self.slice_number_C_LBL.setObjectName(_fromUtf8("slice_number_C_LBL"))
        self.horizontalLayout_11.addWidget(self.slice_number_C_LBL)
        self.slice_C_SB = QtGui.QScrollBar(self.frame_R)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.slice_C_SB.sizePolicy().hasHeightForWidth())
        self.slice_C_SB.setSizePolicy(sizePolicy)
        self.slice_C_SB.setOrientation(QtCore.Qt.Horizontal)
        self.slice_C_SB.setObjectName(_fromUtf8("slice_C_SB"))
        self.horizontalLayout_11.addWidget(self.slice_C_SB)
        self.verticalLayout_2.addLayout(self.horizontalLayout_11)
        self.verticalLayout_4.addWidget(self.splitter)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 721, 25))
        self.menubar.setObjectName(_fromUtf8("menubar"))
        self.menuSerie = QtGui.QMenu(self.menubar)
        self.menuSerie.setObjectName(_fromUtf8("menuSerie"))
        self.menuAction = QtGui.QMenu(self.menubar)
        self.menuAction.setObjectName(_fromUtf8("menuAction"))
        self.menuShow = QtGui.QMenu(self.menubar)
        self.menuShow.setObjectName(_fromUtf8("menuShow"))
        self.menuTools = QtGui.QMenu(self.menubar)
        self.menuTools.setObjectName(_fromUtf8("menuTools"))
        self.menuView = QtGui.QMenu(self.menubar)
        self.menuView.setObjectName(_fromUtf8("menuView"))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtGui.QStatusBar(MainWindow)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        MainWindow.setStatusBar(self.statusbar)
        self.action_load_serie_1 = QtGui.QAction(MainWindow)
        self.action_load_serie_1.setObjectName(_fromUtf8("action_load_serie_1"))
        self.action_load_serie_2 = QtGui.QAction(MainWindow)
        self.action_load_serie_2.setObjectName(_fromUtf8("action_load_serie_2"))
        self.actionRun = QtGui.QAction(MainWindow)
        self.actionRun.setObjectName(_fromUtf8("actionRun"))
        self.actionDelete = QtGui.QAction(MainWindow)
        self.actionDelete.setObjectName(_fromUtf8("actionDelete"))
        self.actionRestart = QtGui.QAction(MainWindow)
        self.actionRestart.setObjectName(_fromUtf8("actionRestart"))
        self.action_color_model = QtGui.QAction(MainWindow)
        self.action_color_model.setObjectName(_fromUtf8("action_color_model"))
        self.action_object_list = QtGui.QAction(MainWindow)
        self.action_object_list.setObjectName(_fromUtf8("action_object_list"))
        self.action_circle = QtGui.QAction(MainWindow)
        self.action_circle.setObjectName(_fromUtf8("action_circle"))
        self.actionRuler = QtGui.QAction(MainWindow)
        self.actionRuler.setObjectName(_fromUtf8("actionRuler"))
        self.actionAxial = QtGui.QAction(MainWindow)
        self.actionAxial.setObjectName(_fromUtf8("actionAxial"))
        self.actionFrontal = QtGui.QAction(MainWindow)
        self.actionFrontal.setObjectName(_fromUtf8("actionFrontal"))
        self.actionSagital = QtGui.QAction(MainWindow)
        self.actionSagital.setObjectName(_fromUtf8("actionSagital"))
        self.menuSerie.addAction(self.action_load_serie_1)
        self.menuSerie.addAction(self.action_load_serie_2)
        self.menuAction.addAction(self.actionRun)
        self.menuAction.addAction(self.actionDelete)
        self.menuAction.addAction(self.actionRestart)
        self.menuShow.addAction(self.action_color_model)
        self.menuShow.addAction(self.action_object_list)
        self.menuTools.addAction(self.action_circle)
        self.menuTools.addAction(self.actionRuler)
        self.menuView.addAction(self.actionAxial)
        self.menuView.addAction(self.actionFrontal)
        self.menuView.addAction(self.actionSagital)
        self.menubar.addAction(self.menuSerie.menuAction())
        self.menubar.addAction(self.menuAction.menuAction())
        self.menubar.addAction(self.menuShow.menuAction())
        self.menubar.addAction(self.menuTools.menuAction())
        self.menubar.addAction(self.menuView.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow", None))
        self.view_L_BTN.setToolTip(_translate("MainWindow", "Show figure", None))
        self.show_im_L_BTN.setToolTip(_translate("MainWindow", "Show data", None))
        self.show_labels_L_BTN.setToolTip(_translate("MainWindow", "Show labels", None))
        self.show_contours_L_BTN.setToolTip(_translate("MainWindow", "Show contours", None))
        self.view_R_BTN.setToolTip(_translate("MainWindow", "Show figure", None))
        self.show_im_R_BTN.setToolTip(_translate("MainWindow", "Show data", None))
        self.show_labels_R_BTN.setToolTip(_translate("MainWindow", "Show labels", None))
        self.show_contours_R_BTN.setToolTip(_translate("MainWindow", "Show contours", None))
        self.slice_number_L_LBL.setText(_translate("MainWindow", "0/0", None))
        self.slice_number_R_LBL.setText(_translate("MainWindow", "0/0", None))
        self.slice_number_C_LBL.setText(_translate("MainWindow", "slice # = 0/0", None))
        self.menuSerie.setTitle(_translate("MainWindow", "Data", None))
        self.menuAction.setTitle(_translate("MainWindow", "Localization", None))
        self.menuShow.setTitle(_translate("MainWindow", "Show", None))
        self.menuTools.setTitle(_translate("MainWindow", "Tools", None))
        self.menuView.setTitle(_translate("MainWindow", "View", None))
        self.action_load_serie_1.setText(_translate("MainWindow", "Load serie #1...", None))
        self.action_load_serie_2.setText(_translate("MainWindow", "Load serie #2...", None))
        self.actionRun.setText(_translate("MainWindow", "Run", None))
        self.actionDelete.setText(_translate("MainWindow", "Delete", None))
        self.actionRestart.setText(_translate("MainWindow", "Restart", None))
        self.action_color_model.setText(_translate("MainWindow", "Color Model", None))
        self.action_object_list.setText(_translate("MainWindow", "Object List", None))
        self.action_circle.setText(_translate("MainWindow", "Circle", None))
        self.actionRuler.setText(_translate("MainWindow", "Ruler", None))
        self.actionAxial.setText(_translate("MainWindow", "Axial", None))
        self.actionFrontal.setText(_translate("MainWindow", "Frontal", None))
        self.actionSagital.setText(_translate("MainWindow", "Sagital", None))

