# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'hist_widget_GUI.ui'
#
# Created: Wed Sep 16 07:24:34 2015
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
        Form.resize(528, 574)
        Form.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.verticalLayout = QtGui.QVBoxLayout(Form)
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.histogram_F = QtGui.QFrame(Form)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.histogram_F.sizePolicy().hasHeightForWidth())
        self.histogram_F.setSizePolicy(sizePolicy)
        self.histogram_F.setMinimumSize(QtCore.QSize(0, 100))
        self.histogram_F.setFrameShape(QtGui.QFrame.Box)
        self.histogram_F.setObjectName(_fromUtf8("histogram_F"))
        self.verticalLayout.addWidget(self.histogram_F)
        self.verticalLayout_2 = QtGui.QVBoxLayout()
        self.verticalLayout_2.setObjectName(_fromUtf8("verticalLayout_2"))
        self.frame = QtGui.QFrame(Form)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame.sizePolicy().hasHeightForWidth())
        self.frame.setSizePolicy(sizePolicy)
        self.frame.setMinimumSize(QtCore.QSize(0, 50))
        self.frame.setFrameShape(QtGui.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtGui.QFrame.Raised)
        self.frame.setObjectName(_fromUtf8("frame"))
        self.verticalLayout_4 = QtGui.QVBoxLayout(self.frame)
        self.verticalLayout_4.setObjectName(_fromUtf8("verticalLayout_4"))
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.label_12 = QtGui.QLabel(self.frame)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_12.sizePolicy().hasHeightForWidth())
        self.label_12.setSizePolicy(sizePolicy)
        self.label_12.setMinimumSize(QtCore.QSize(0, 0))
        self.label_12.setMaximumSize(QtCore.QSize(90, 16777215))
        self.label_12.setTextFormat(QtCore.Qt.RichText)
        self.label_12.setObjectName(_fromUtf8("label_12"))
        self.horizontalLayout.addWidget(self.label_12)
        self.label = QtGui.QLabel(self.frame)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
        self.label.setTextFormat(QtCore.Qt.RichText)
        self.label.setObjectName(_fromUtf8("label"))
        self.horizontalLayout.addWidget(self.label)
        self.hypo_mean_LE = QtGui.QLineEdit(self.frame)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.hypo_mean_LE.sizePolicy().hasHeightForWidth())
        self.hypo_mean_LE.setSizePolicy(sizePolicy)
        self.hypo_mean_LE.setMaximumSize(QtCore.QSize(50, 16777215))
        self.hypo_mean_LE.setObjectName(_fromUtf8("hypo_mean_LE"))
        self.horizontalLayout.addWidget(self.hypo_mean_LE)
        self.hypo_mean_SL = QtGui.QSlider(self.frame)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.hypo_mean_SL.sizePolicy().hasHeightForWidth())
        self.hypo_mean_SL.setSizePolicy(sizePolicy)
        self.hypo_mean_SL.setMaximum(100)
        self.hypo_mean_SL.setProperty("value", 0)
        self.hypo_mean_SL.setOrientation(QtCore.Qt.Horizontal)
        self.hypo_mean_SL.setObjectName(_fromUtf8("hypo_mean_SL"))
        self.horizontalLayout.addWidget(self.hypo_mean_SL)
        self.label_2 = QtGui.QLabel(self.frame)
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.horizontalLayout.addWidget(self.label_2)
        self.hypo_std_SB = QtGui.QSpinBox(self.frame)
        self.hypo_std_SB.setMinimum(1)
        self.hypo_std_SB.setObjectName(_fromUtf8("hypo_std_SB"))
        self.horizontalLayout.addWidget(self.hypo_std_SB)
        self.verticalLayout_4.addLayout(self.horizontalLayout)
        self.verticalLayout_2.addWidget(self.frame)
        self.frame_2 = QtGui.QFrame(Form)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_2.sizePolicy().hasHeightForWidth())
        self.frame_2.setSizePolicy(sizePolicy)
        self.frame_2.setMinimumSize(QtCore.QSize(0, 50))
        self.frame_2.setFrameShape(QtGui.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtGui.QFrame.Raised)
        self.frame_2.setObjectName(_fromUtf8("frame_2"))
        self.verticalLayout_7 = QtGui.QVBoxLayout(self.frame_2)
        self.verticalLayout_7.setObjectName(_fromUtf8("verticalLayout_7"))
        self.horizontalLayout_3 = QtGui.QHBoxLayout()
        self.horizontalLayout_3.setObjectName(_fromUtf8("horizontalLayout_3"))
        self.label_13 = QtGui.QLabel(self.frame_2)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_13.sizePolicy().hasHeightForWidth())
        self.label_13.setSizePolicy(sizePolicy)
        self.label_13.setMaximumSize(QtCore.QSize(90, 16777215))
        self.label_13.setObjectName(_fromUtf8("label_13"))
        self.horizontalLayout_3.addWidget(self.label_13)
        self.label_14 = QtGui.QLabel(self.frame_2)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_14.sizePolicy().hasHeightForWidth())
        self.label_14.setSizePolicy(sizePolicy)
        self.label_14.setObjectName(_fromUtf8("label_14"))
        self.horizontalLayout_3.addWidget(self.label_14)
        self.heal_mean_LE = QtGui.QLineEdit(self.frame_2)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.heal_mean_LE.sizePolicy().hasHeightForWidth())
        self.heal_mean_LE.setSizePolicy(sizePolicy)
        self.heal_mean_LE.setMaximumSize(QtCore.QSize(50, 16777215))
        self.heal_mean_LE.setObjectName(_fromUtf8("heal_mean_LE"))
        self.horizontalLayout_3.addWidget(self.heal_mean_LE)
        self.heal_mean_SL = QtGui.QSlider(self.frame_2)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.heal_mean_SL.sizePolicy().hasHeightForWidth())
        self.heal_mean_SL.setSizePolicy(sizePolicy)
        self.heal_mean_SL.setMaximum(100)
        self.heal_mean_SL.setProperty("value", 0)
        self.heal_mean_SL.setOrientation(QtCore.Qt.Horizontal)
        self.heal_mean_SL.setObjectName(_fromUtf8("heal_mean_SL"))
        self.horizontalLayout_3.addWidget(self.heal_mean_SL)
        self.label_15 = QtGui.QLabel(self.frame_2)
        self.label_15.setObjectName(_fromUtf8("label_15"))
        self.horizontalLayout_3.addWidget(self.label_15)
        self.heal_std_SB = QtGui.QSpinBox(self.frame_2)
        self.heal_std_SB.setMinimum(1)
        self.heal_std_SB.setObjectName(_fromUtf8("heal_std_SB"))
        self.horizontalLayout_3.addWidget(self.heal_std_SB)
        self.verticalLayout_7.addLayout(self.horizontalLayout_3)
        self.verticalLayout_2.addWidget(self.frame_2)
        self.frame_6 = QtGui.QFrame(Form)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_6.sizePolicy().hasHeightForWidth())
        self.frame_6.setSizePolicy(sizePolicy)
        self.frame_6.setMinimumSize(QtCore.QSize(0, 50))
        self.frame_6.setFrameShape(QtGui.QFrame.StyledPanel)
        self.frame_6.setFrameShadow(QtGui.QFrame.Raised)
        self.frame_6.setObjectName(_fromUtf8("frame_6"))
        self.verticalLayout_8 = QtGui.QVBoxLayout(self.frame_6)
        self.verticalLayout_8.setObjectName(_fromUtf8("verticalLayout_8"))
        self.horizontalLayout_10 = QtGui.QHBoxLayout()
        self.horizontalLayout_10.setObjectName(_fromUtf8("horizontalLayout_10"))
        self.label_16 = QtGui.QLabel(self.frame_6)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_16.sizePolicy().hasHeightForWidth())
        self.label_16.setSizePolicy(sizePolicy)
        self.label_16.setMaximumSize(QtCore.QSize(90, 16777215))
        self.label_16.setObjectName(_fromUtf8("label_16"))
        self.horizontalLayout_10.addWidget(self.label_16)
        self.label_17 = QtGui.QLabel(self.frame_6)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_17.sizePolicy().hasHeightForWidth())
        self.label_17.setSizePolicy(sizePolicy)
        self.label_17.setObjectName(_fromUtf8("label_17"))
        self.horizontalLayout_10.addWidget(self.label_17)
        self.hyper_mean_LE = QtGui.QLineEdit(self.frame_6)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.hyper_mean_LE.sizePolicy().hasHeightForWidth())
        self.hyper_mean_LE.setSizePolicy(sizePolicy)
        self.hyper_mean_LE.setMaximumSize(QtCore.QSize(50, 16777215))
        self.hyper_mean_LE.setObjectName(_fromUtf8("hyper_mean_LE"))
        self.horizontalLayout_10.addWidget(self.hyper_mean_LE)
        self.hyper_mean_SL = QtGui.QSlider(self.frame_6)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.hyper_mean_SL.sizePolicy().hasHeightForWidth())
        self.hyper_mean_SL.setSizePolicy(sizePolicy)
        self.hyper_mean_SL.setMaximum(100)
        self.hyper_mean_SL.setProperty("value", 0)
        self.hyper_mean_SL.setOrientation(QtCore.Qt.Horizontal)
        self.hyper_mean_SL.setObjectName(_fromUtf8("hyper_mean_SL"))
        self.horizontalLayout_10.addWidget(self.hyper_mean_SL)
        self.label_18 = QtGui.QLabel(self.frame_6)
        self.label_18.setObjectName(_fromUtf8("label_18"))
        self.horizontalLayout_10.addWidget(self.label_18)
        self.hyper_std_SB = QtGui.QSpinBox(self.frame_6)
        self.hyper_std_SB.setMinimum(1)
        self.hyper_std_SB.setObjectName(_fromUtf8("hyper_std_SB"))
        self.horizontalLayout_10.addWidget(self.hyper_std_SB)
        self.verticalLayout_8.addLayout(self.horizontalLayout_10)
        self.verticalLayout_2.addWidget(self.frame_6)
        self.verticalLayout.addLayout(self.verticalLayout_2)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        Form.setWindowTitle(_translate("Form", "Form", None))
        self.label_12.setText(_translate("Form", "<b>Hypodense:</b>", None))
        self.label.setText(_translate("Form", "mean=", None))
        self.hypo_mean_LE.setText(_translate("Form", "0", None))
        self.label_2.setText(_translate("Form", "std=", None))
        self.label_13.setText(_translate("Form", "<b>Healthy:</b>", None))
        self.label_14.setText(_translate("Form", "mean=", None))
        self.heal_mean_LE.setText(_translate("Form", "0", None))
        self.label_15.setText(_translate("Form", "std=", None))
        self.label_16.setText(_translate("Form", "<b>Hyperdense:</b>", None))
        self.label_17.setText(_translate("Form", "mean=", None))
        self.hyper_mean_LE.setText(_translate("Form", "0", None))
        self.label_18.setText(_translate("Form", "std=", None))

