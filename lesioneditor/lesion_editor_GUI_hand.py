__author__ = 'tomas'

from PyQt4 import QtCore, QtGui
import sys
import os

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


class LesionEditorGUI(object):
    # def __init__(self):
    #     QtGui.QWidget.__init__(self)
    #     self.setupUi()

    def setupUi(self, MainWindow):

        cur_path = os.path.abspath(__file__)
        head, tail = os.path.split(cur_path)

        MainWindow.resize(100, 100)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        MainWindow.setSizePolicy(sizePolicy)

        self.centralwidget = QtGui.QWidget()
        self.centralwidget.setLayoutDirection(QtCore.Qt.LeftToRight)
        MainWindow.setCentralWidget(self.centralwidget)

        # -- BUTTONS --
        # left view
        self.view_L_BTN = QtGui.QPushButton()
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        self.view_L_BTN.setSizePolicy(sizePolicy)
        self.view_L_BTN.setMinimumSize(QtCore.QSize(22, 22))
        self.view_L_BTN.setMaximumSize(QtCore.QSize(22, 22))
        self.view_L_BTN.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        # self.view_L_BTN.setToolTip(_fromUtf8("Enable / disable view window"))
        self.view_L_BTN.setToolTip('Enable / disable view window')
        self.view_L_BTN.setIcon(QtGui.QIcon(os.path.join(head, 'icons', 'Eye.png')))
        self.view_L_BTN.setIconSize(QtCore.QSize(16, 16))

        self.show_im_L_BTN = QtGui.QPushButton()
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        self.show_im_L_BTN.setSizePolicy(sizePolicy)
        self.show_im_L_BTN.setMinimumSize(QtCore.QSize(22, 22))
        self.show_im_L_BTN.setMaximumSize(QtCore.QSize(22, 22))
        self.show_im_L_BTN.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.show_im_L_BTN.setToolTip(_fromUtf8("Show data"))
        self.show_im_L_BTN.setIcon(QtGui.QIcon(os.path.join(head, 'icons', 'Stock graph.png')))
        self.show_im_L_BTN.setIconSize(QtCore.QSize(16, 16))

        self.show_labels_L_BTN = QtGui.QPushButton()
        self.show_labels_L_BTN.setEnabled(False)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        self.show_labels_L_BTN.setSizePolicy(sizePolicy)
        self.show_labels_L_BTN.setMinimumSize(QtCore.QSize(22, 22))
        self.show_labels_L_BTN.setMaximumSize(QtCore.QSize(22, 22))
        self.show_labels_L_BTN.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.show_labels_L_BTN.setToolTip(_fromUtf8("Show labels"))
        self.show_labels_L_BTN.setIcon(QtGui.QIcon(os.path.join(head, 'icons', 'Blue tag.png')))
        self.show_labels_L_BTN.setIconSize(QtCore.QSize(16, 16))

        self.show_contours_L_BTN = QtGui.QPushButton()
        self.show_contours_L_BTN.setEnabled(False)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        self.show_contours_L_BTN.setSizePolicy(sizePolicy)
        self.show_contours_L_BTN.setMinimumSize(QtCore.QSize(22, 22))
        self.show_contours_L_BTN.setMaximumSize(QtCore.QSize(22, 22))
        self.show_contours_L_BTN.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.show_contours_L_BTN.setToolTip(_fromUtf8("Show contours"))
        self.show_contours_L_BTN.setIcon(QtGui.QIcon(os.path.join(head, 'icons', 'Brush.png')))
        self.show_contours_L_BTN.setIconSize(QtCore.QSize(16, 16))

        # right view
        self.view_R_BTN = QtGui.QPushButton()
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        self.view_R_BTN.setSizePolicy(sizePolicy)
        self.view_R_BTN.setMinimumSize(QtCore.QSize(22, 22))
        self.view_R_BTN.setMaximumSize(QtCore.QSize(22, 22))
        self.view_R_BTN.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.view_R_BTN.setToolTip(_fromUtf8("Enable / disable view window"))
        self.view_R_BTN.setIcon(QtGui.QIcon(os.path.join(head, 'icons', 'Eye.png')))
        self.view_R_BTN.setIconSize(QtCore.QSize(16, 16))

        self.show_im_R_BTN = QtGui.QPushButton()
        self.show_im_R_BTN.setEnabled(False)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        self.show_im_R_BTN.setSizePolicy(sizePolicy)
        self.show_im_R_BTN.setMinimumSize(QtCore.QSize(22, 22))
        self.show_im_R_BTN.setMaximumSize(QtCore.QSize(22, 22))
        self.show_im_R_BTN.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.show_im_R_BTN.setToolTip(_fromUtf8("Show data"))
        self.show_im_R_BTN.setIcon(QtGui.QIcon(os.path.join(head, 'icons', 'Stock graph.png')))
        self.show_im_R_BTN.setIconSize(QtCore.QSize(16, 16))

        self.show_labels_R_BTN = QtGui.QPushButton()
        self.show_labels_R_BTN.setEnabled(False)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        self.show_labels_R_BTN.setSizePolicy(sizePolicy)
        self.show_labels_R_BTN.setMinimumSize(QtCore.QSize(22, 22))
        self.show_labels_R_BTN.setMaximumSize(QtCore.QSize(22, 22))
        self.show_labels_R_BTN.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.show_labels_R_BTN.setToolTip(_fromUtf8("Show labels"))
        self.show_labels_R_BTN.setIcon(QtGui.QIcon(os.path.join(head, 'icons', 'Blue tag.png')))
        self.show_labels_R_BTN.setIconSize(QtCore.QSize(16, 16))

        self.show_contours_R_BTN = QtGui.QPushButton()
        self.show_contours_R_BTN.setEnabled(False)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        self.show_contours_R_BTN.setSizePolicy(sizePolicy)
        self.show_contours_R_BTN.setMinimumSize(QtCore.QSize(22, 22))
        self.show_contours_R_BTN.setMaximumSize(QtCore.QSize(22, 22))
        self.show_contours_R_BTN.setCursor(QtGui.QCursor(QtCore.Qt.PointingHandCursor))
        self.show_contours_R_BTN.setToolTip(_fromUtf8("Show contours"))
        self.show_contours_R_BTN.setIcon(QtGui.QIcon(os.path.join(head, 'icons', 'Brush.png')))
        self.show_contours_R_BTN.setIconSize(QtCore.QSize(16, 16))

        self.line = QtGui.QFrame()
        self.line.setFrameShape(QtGui.QFrame.VLine)
        self.line.setFrameShadow(QtGui.QFrame.Sunken)

        # -- COMBO BOXES --
        self.figure_L_CB = QtGui.QComboBox()
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        self.figure_L_CB.setSizePolicy(sizePolicy)
        self.figure_L_CB.setMinimumSize(QtCore.QSize(50, 0))

        self.figure_R_CB = QtGui.QComboBox()
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        self.figure_R_CB.setSizePolicy(sizePolicy)
        self.figure_R_CB.setMinimumSize(QtCore.QSize(50, 0))

        # -- VIEWER --
        self.viewer_F = QtGui.QFrame()
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        self.viewer_F.setSizePolicy(sizePolicy)
        self.viewer_F.setMinimumSize(QtCore.QSize(0, 0))
        self.viewer_F.setFrameShape(QtGui.QFrame.Box)

        # -- SCROLLBARS --
        self.slice_L_SB = QtGui.QScrollBar()
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        self.slice_L_SB.setSizePolicy(sizePolicy)
        self.slice_L_SB.setMinimumSize(QtCore.QSize(50, 12))
        self.slice_L_SB.setMaximumSize(QtCore.QSize(100, 12))
        self.slice_L_SB.setOrientation(QtCore.Qt.Horizontal)

        self.slice_C_SB = QtGui.QScrollBar()
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        self.slice_C_SB.setSizePolicy(sizePolicy)
        # self.slice_C_SB.setMaximumSize(QtCore.QSize(150, 16777215))
        self.slice_C_SB.setMinimumHeight(12)
        self.slice_C_SB.setMaximumHeight(12)
        self.slice_C_SB.setOrientation(QtCore.Qt.Horizontal)

        self.slice_R_SB = QtGui.QScrollBar()
        # self.slice_R_SB.setEnabled(True)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.MinimumExpanding, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        self.slice_R_SB.setSizePolicy(sizePolicy)
        self.slice_R_SB.setMinimumSize(QtCore.QSize(50, 12))
        self.slice_R_SB.setMaximumSize(QtCore.QSize(100, 12))
        self.slice_R_SB.setOrientation(QtCore.Qt.Horizontal)

        # -- LABELS --
        self.slice_number_L_LBL = QtGui.QLabel()
        self.slice_number_L_LBL.setMinimumSize(QtCore.QSize(40, 40))
        self.slice_number_L_LBL.setMaximumSize(QtCore.QSize(40, 40))
        self.slice_number_L_LBL.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.slice_number_L_LBL.setText("0/0")

        self.slice_number_R_LBL = QtGui.QLabel()
        self.slice_number_R_LBL.setMinimumSize(QtCore.QSize(40, 40))
        self.slice_number_R_LBL.setMaximumSize(QtCore.QSize(40, 40))
        self.slice_number_R_LBL.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.slice_number_R_LBL.setText("0/0")

        # -- MENU --
        self.menubar = QtGui.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 815, 25))
        self.menuSerie = QtGui.QMenu('Data')
        self.menuAction = QtGui.QMenu('Action')
        self.menuShow = QtGui.QMenu('Show')
        self.menuTools = QtGui.QMenu('Tools')
        # self.menuView = QtGui.QMenu('View')
        MainWindow.setMenuBar(self.menubar)

        self.action_load_serie_1 = QtGui.QAction('Load serie #1...', MainWindow)
        self.action_load_serie_2 = QtGui.QAction('Load serie #2...', MainWindow)
        self.action_run = QtGui.QAction('Run localization', MainWindow)
        # self.action_delete = QtGui.QAction('Delete localization', MainWindow)
        # self.action_restart = QtGui.QAction('Restart localization', MainWindow)
        self.action_show_color_model = QtGui.QAction('Color Model', MainWindow)
        self.action_show_object_list = QtGui.QAction('Object List', MainWindow)
        self.action_circle = QtGui.QAction('Circle', MainWindow)
        self.actionRuler = QtGui.QAction('Ruler', MainWindow)
        # self.actionAxial = QtGui.QAction('Axial', MainWindow)
        # self.actionFrontal = QtGui.QAction('Frontal', MainWindow)
        # self.actionSagital = QtGui.QAction('Sagital', MainWindow)
        self.action_calculate_color_model = QtGui.QAction('Calculate color model', MainWindow)
        self.menuSerie.addAction(self.action_load_serie_1)
        self.menuSerie.addAction(self.action_load_serie_2)
        self.menuAction.addAction(self.action_calculate_color_model)
        self.menuAction.addSeparator()
        self.menuAction.addAction(self.action_run)
        # self.menuAction.addAction(self.action_delete)
        # self.menuAction.addAction(self.action_restart)
        self.menuShow.addAction(self.action_show_color_model)
        self.menuShow.addAction(self.action_show_object_list)
        self.menuTools.addAction(self.action_circle)
        self.menuTools.addAction(self.actionRuler)
        # self.menuView.addAction(self.actionAxial)
        # self.menuView.addAction(self.actionFrontal)
        # self.menuView.addAction(self.actionSagital)
        self.menubar.addAction(self.menuSerie.menuAction())
        self.menubar.addAction(self.menuAction.menuAction())
        self.menubar.addAction(self.menuShow.menuAction())
        self.menubar.addAction(self.menuTools.menuAction())
        # self.menubar.addAction(self.menuView.menuAction())


        # -- ADDING WIDGETS AND LAYOUTS --
        main_layout = QtGui.QVBoxLayout()  # main vertical layout
        view_layout = QtGui.QHBoxLayout()  # contains buttons for changing view
        scroll_layout = QtGui.QHBoxLayout()  # contains scrollbars for navigating thru data

        view_layout.addWidget(self.view_L_BTN)
        view_layout.addWidget(self.show_im_L_BTN)
        view_layout.addWidget(self.show_labels_L_BTN)
        view_layout.addWidget(self.show_contours_L_BTN)
        view_layout.addWidget(self.figure_L_CB)
        view_layout.addWidget(self.line)
        view_layout.addWidget(self.figure_R_CB)
        view_layout.addWidget(self.view_R_BTN)
        view_layout.addWidget(self.show_im_R_BTN)
        view_layout.addWidget(self.show_labels_R_BTN)
        view_layout.addWidget(self.show_contours_R_BTN)

        scroll_layout.addWidget(self.slice_number_L_LBL)
        scroll_layout.addWidget(self.slice_L_SB)
        scroll_layout.addWidget(self.slice_C_SB)
        scroll_layout.addWidget(self.slice_R_SB)
        scroll_layout.addWidget(self.slice_number_R_LBL)

        main_layout.addLayout(view_layout)
        main_layout.addWidget(self.viewer_F)
        main_layout.addLayout(scroll_layout)
        self.centralwidget.setLayout(main_layout)

    # def retranslateUi(self, MainWindow):
        # MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow", None))
        # self.view_L_BTN.setToolTip(_translate("MainWindow", "Show figure", None))
        # self.show_im_L_BTN.setToolTip(_translate("MainWindow", "Show data", None))
        # self.show_labels_L_BTN.setToolTip(_translate("MainWindow", "Show labels", None))
        # self.show_contours_L_BTN.setToolTip(_translate("MainWindow", "Show contours", None))
        # self.view_R_BTN.setToolTip(_translate("MainWindow", "Show figure", None))
        # self.show_im_R_BTN.setToolTip(_translate("MainWindow", "Show data", None))
        # self.show_labels_R_BTN.setToolTip(_translate("MainWindow", "Show labels", None))
        # self.show_contours_R_BTN.setToolTip(_translate("MainWindow", "Show contours", None))
        # self.slice_number_L_LBL.setText(_translate("MainWindow", "0/0", None))
        # self.slice_number_R_LBL.setText(_translate("MainWindow", "0/0", None))
        # self.slice_number_C_LBL.setText(_translate("MainWindow", "slice # = 0/0", None))
        # self.menuSerie.setTitle(_translate("MainWindow", "Data", None))
        # self.menuAction.setTitle(_translate("MainWindow", "Localization", None))
        # self.menuShow.setTitle(_translate("MainWindow", "Show", None))
        # self.menuTools.setTitle(_translate("MainWindow", "Tools", None))
        # self.menuView.setTitle(_translate("MainWindow", "View", None))
        # self.action_Load_serie_1.setText(_translate("MainWindow", "Load serie #1...", None))
        # self.action_Load_serie_2.setText(_translate("MainWindow", "Load serie #2...", None))
        # self.actionRun.setText(_translate("MainWindow", "Run", None))
        # self.actionDelete.setText(_translate("MainWindow", "Delete", None))
        # self.actionRestart.setText(_translate("MainWindow", "Restart", None))
        # self.actionColor_Model.setText(_translate("MainWindow", "Color Model", None))
        # self.actionObject_List.setText(_translate("MainWindow", "Object List", None))
        # self.actionCircle.setText(_translate("MainWindow", "Circle", None))
        # self.actionRuler.setText(_translate("MainWindow", "Ruler", None))
        # self.actionAxial.setText(_translate("MainWindow", "Axial", None))
        # self.actionFrontal.setText(_translate("MainWindow", "Frontal", None))
        # self.actionSagital.setText(_translate("MainWindow", "Sagital", None))


if __name__ == '__main__':

    class editor(QtGui.QMainWindow):
        def __init__(self):
            QtGui.QWidget.__init__(self, parent=None)
            self.ui = LesionEditorGUI()
            self.ui.setupUi(self)

    app = QtGui.QApplication(sys.argv)
    e = editor()
    e.show()
    sys.exit(app.exec_())
