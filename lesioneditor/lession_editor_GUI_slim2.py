# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'lession_editor_GUI_slim2.ui'
#
# Created: Tue Sep  8 10:43:56 2015
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
        MainWindow.resize(720, 839)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
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
        self.viewer_F = QtGui.QFrame(self.centralwidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.viewer_F.sizePolicy().hasHeightForWidth())
        self.viewer_F.setSizePolicy(sizePolicy)
        self.viewer_F.setMinimumSize(QtCore.QSize(0, 0))
        self.viewer_F.setFrameShape(QtGui.QFrame.Box)
        self.viewer_F.setObjectName(_fromUtf8("viewer_F"))
        self.verticalLayout_4.addWidget(self.viewer_F)
        MainWindow.setCentralWidget(self.centralwidget)
        self.action_Load_serie_1 = QtGui.QAction(MainWindow)
        self.action_Load_serie_1.setObjectName(_fromUtf8("action_Load_serie_1"))
        self.action_Load_serie_2 = QtGui.QAction(MainWindow)
        self.action_Load_serie_2.setObjectName(_fromUtf8("action_Load_serie_2"))
        self.actionRun = QtGui.QAction(MainWindow)
        self.actionRun.setObjectName(_fromUtf8("actionRun"))
        self.actionDelete = QtGui.QAction(MainWindow)
        self.actionDelete.setObjectName(_fromUtf8("actionDelete"))
        self.actionRestart = QtGui.QAction(MainWindow)
        self.actionRestart.setObjectName(_fromUtf8("actionRestart"))
        self.actionColor_Model = QtGui.QAction(MainWindow)
        self.actionColor_Model.setObjectName(_fromUtf8("actionColor_Model"))
        self.actionObject_List = QtGui.QAction(MainWindow)
        self.actionObject_List.setObjectName(_fromUtf8("actionObject_List"))
        self.actionCircle = QtGui.QAction(MainWindow)
        self.actionCircle.setObjectName(_fromUtf8("actionCircle"))
        self.actionRuler = QtGui.QAction(MainWindow)
        self.actionRuler.setObjectName(_fromUtf8("actionRuler"))
        self.actionAxial = QtGui.QAction(MainWindow)
        self.actionAxial.setObjectName(_fromUtf8("actionAxial"))
        self.actionFrontal = QtGui.QAction(MainWindow)
        self.actionFrontal.setObjectName(_fromUtf8("actionFrontal"))
        self.actionSagital = QtGui.QAction(MainWindow)
        self.actionSagital.setObjectName(_fromUtf8("actionSagital"))

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow", None))
        self.action_Load_serie_1.setText(_translate("MainWindow", "Load serie #1...", None))
        self.action_Load_serie_2.setText(_translate("MainWindow", "Load serie #2...", None))
        self.actionRun.setText(_translate("MainWindow", "Run", None))
        self.actionDelete.setText(_translate("MainWindow", "Delete", None))
        self.actionRestart.setText(_translate("MainWindow", "Restart", None))
        self.actionColor_Model.setText(_translate("MainWindow", "Color Model", None))
        self.actionObject_List.setText(_translate("MainWindow", "Object List", None))
        self.actionCircle.setText(_translate("MainWindow", "Circle", None))
        self.actionRuler.setText(_translate("MainWindow", "Ruler", None))
        self.actionAxial.setText(_translate("MainWindow", "Axial", None))
        self.actionFrontal.setText(_translate("MainWindow", "Frontal", None))
        self.actionSagital.setText(_translate("MainWindow", "Sagital", None))

