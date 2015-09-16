# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'objects_widget_GUI.ui'
#
# Created: Wed Sep 16 09:38:17 2015
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

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName(_fromUtf8("Form"))
        Form.resize(400, 346)
        self.verticalLayout = QtGui.QVBoxLayout(Form)
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.area_layout = QtGui.QHBoxLayout()
        self.area_layout.setObjectName(_fromUtf8("area_layout"))
        self.label = QtGui.QLabel(Form)
        self.label.setObjectName(_fromUtf8("label"))
        self.area_layout.addWidget(self.label)
        self.verticalLayout.addLayout(self.area_layout)
        self.horizontalLayout_2 = QtGui.QHBoxLayout()
        self.horizontalLayout_2.setObjectName(_fromUtf8("horizontalLayout_2"))
        self.label_2 = QtGui.QLabel(Form)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_2.sizePolicy().hasHeightForWidth())
        self.label_2.setSizePolicy(sizePolicy)
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.horizontalLayout_2.addWidget(self.label_2)
        self.lineEdit = QtGui.QLineEdit(Form)
        self.lineEdit.setMaximumSize(QtCore.QSize(50, 16777215))
        self.lineEdit.setObjectName(_fromUtf8("lineEdit"))
        self.horizontalLayout_2.addWidget(self.lineEdit)
        self.horizontalSlider = QtGui.QSlider(Form)
        self.horizontalSlider.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider.setObjectName(_fromUtf8("horizontalSlider"))
        self.horizontalLayout_2.addWidget(self.horizontalSlider)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.density_layout = QtGui.QHBoxLayout()
        self.density_layout.setObjectName(_fromUtf8("density_layout"))
        self.label_3 = QtGui.QLabel(Form)
        self.label_3.setObjectName(_fromUtf8("label_3"))
        self.density_layout.addWidget(self.label_3)
        self.verticalLayout.addLayout(self.density_layout)
        self.objects_TV = QtGui.QTableView(Form)
        self.objects_TV.setEditTriggers(QtGui.QAbstractItemView.NoEditTriggers)
        self.objects_TV.setSelectionBehavior(QtGui.QAbstractItemView.SelectRows)
        self.objects_TV.setObjectName(_fromUtf8("objects_TV"))
        self.verticalLayout.addWidget(self.objects_TV)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        Form.setWindowTitle(_translate("Form", "Objects", None))
        self.label.setText(_translate("Form", "area [ml]:", None))
        self.label_2.setText(_translate("Form", "min. compactness=", None))
        self.label_3.setText(_translate("Form", "density [HU]:", None))

