# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'lession_editor_GUI.ui'
#
# Created: Tue Sep 23 10:56:31 2014
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
        MainWindow.resize(1078, 808)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.verticalLayout_4 = QtGui.QVBoxLayout(self.centralwidget)
        self.verticalLayout_4.setObjectName(_fromUtf8("verticalLayout_4"))
        self.splitter = QtGui.QSplitter(self.centralwidget)
        self.splitter.setOrientation(QtCore.Qt.Horizontal)
        self.splitter.setObjectName(_fromUtf8("splitter"))
        self.frame = QtGui.QFrame(self.splitter)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame.sizePolicy().hasHeightForWidth())
        self.frame.setSizePolicy(sizePolicy)
        self.frame.setMinimumSize(QtCore.QSize(0, 0))
        self.frame.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.frame.setBaseSize(QtCore.QSize(0, 0))
        self.frame.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.frame.setFrameShape(QtGui.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtGui.QFrame.Raised)
        self.frame.setLineWidth(1)
        self.frame.setObjectName(_fromUtf8("frame"))
        self.verticalLayout = QtGui.QVBoxLayout(self.frame)
        self.verticalLayout.setObjectName(_fromUtf8("verticalLayout"))
        self.label = QtGui.QLabel(self.frame)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
        self.label.setMaximumSize(QtCore.QSize(16777215, 30))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setFrameShape(QtGui.QFrame.WinPanel)
        self.label.setTextFormat(QtCore.Qt.PlainText)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName(_fromUtf8("label"))
        self.verticalLayout.addWidget(self.label)
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.auto_colormodel_BTN = QtGui.QPushButton(self.frame)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.auto_colormodel_BTN.sizePolicy().hasHeightForWidth())
        self.auto_colormodel_BTN.setSizePolicy(sizePolicy)
        self.auto_colormodel_BTN.setMaximumSize(QtCore.QSize(100, 16777215))
        self.auto_colormodel_BTN.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.auto_colormodel_BTN.setObjectName(_fromUtf8("auto_colormodel_BTN"))
        self.horizontalLayout.addWidget(self.auto_colormodel_BTN)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.histogram_WDT = QtGui.QWidget(self.frame)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.histogram_WDT.sizePolicy().hasHeightForWidth())
        self.histogram_WDT.setSizePolicy(sizePolicy)
        self.histogram_WDT.setMinimumSize(QtCore.QSize(0, 100))
        self.histogram_WDT.setObjectName(_fromUtf8("histogram_WDT"))
        self.verticalLayout.addWidget(self.histogram_WDT)
        self.horizontalLayout_4 = QtGui.QHBoxLayout()
        self.horizontalLayout_4.setObjectName(_fromUtf8("horizontalLayout_4"))
        self.frame_3 = QtGui.QFrame(self.frame)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_3.sizePolicy().hasHeightForWidth())
        self.frame_3.setSizePolicy(sizePolicy)
        self.frame_3.setMinimumSize(QtCore.QSize(0, 50))
        self.frame_3.setFrameShape(QtGui.QFrame.StyledPanel)
        self.frame_3.setFrameShadow(QtGui.QFrame.Raised)
        self.frame_3.setObjectName(_fromUtf8("frame_3"))
        self.verticalLayout_3 = QtGui.QVBoxLayout(self.frame_3)
        self.verticalLayout_3.setObjectName(_fromUtf8("verticalLayout_3"))
        self.label_3 = QtGui.QLabel(self.frame_3)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_3.sizePolicy().hasHeightForWidth())
        self.label_3.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_3.setFont(font)
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName(_fromUtf8("label_3"))
        self.verticalLayout_3.addWidget(self.label_3)
        self.line = QtGui.QFrame(self.frame_3)
        self.line.setFrameShape(QtGui.QFrame.HLine)
        self.line.setFrameShadow(QtGui.QFrame.Sunken)
        self.line.setObjectName(_fromUtf8("line"))
        self.verticalLayout_3.addWidget(self.line)
        self.horizontalLayout_2 = QtGui.QHBoxLayout()
        self.horizontalLayout_2.setObjectName(_fromUtf8("horizontalLayout_2"))
        self.label_6 = QtGui.QLabel(self.frame_3)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_6.sizePolicy().hasHeightForWidth())
        self.label_6.setSizePolicy(sizePolicy)
        self.label_6.setAlignment(QtCore.Qt.AlignCenter)
        self.label_6.setObjectName(_fromUtf8("label_6"))
        self.horizontalLayout_2.addWidget(self.label_6)
        self.hypo_mean_SB = QtGui.QSpinBox(self.frame_3)
        self.hypo_mean_SB.setMaximum(256)
        self.hypo_mean_SB.setObjectName(_fromUtf8("hypo_mean_SB"))
        self.horizontalLayout_2.addWidget(self.hypo_mean_SB)
        self.verticalLayout_3.addLayout(self.horizontalLayout_2)
        self.horizontalLayout_5 = QtGui.QHBoxLayout()
        self.horizontalLayout_5.setObjectName(_fromUtf8("horizontalLayout_5"))
        self.label_7 = QtGui.QLabel(self.frame_3)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_7.sizePolicy().hasHeightForWidth())
        self.label_7.setSizePolicy(sizePolicy)
        self.label_7.setAlignment(QtCore.Qt.AlignCenter)
        self.label_7.setObjectName(_fromUtf8("label_7"))
        self.horizontalLayout_5.addWidget(self.label_7)
        self.hypo_std_SB = QtGui.QSpinBox(self.frame_3)
        self.hypo_std_SB.setObjectName(_fromUtf8("hypo_std_SB"))
        self.horizontalLayout_5.addWidget(self.hypo_std_SB)
        self.verticalLayout_3.addLayout(self.horizontalLayout_5)
        self.horizontalLayout_4.addWidget(self.frame_3)
        self.frame_4 = QtGui.QFrame(self.frame)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_4.sizePolicy().hasHeightForWidth())
        self.frame_4.setSizePolicy(sizePolicy)
        self.frame_4.setMinimumSize(QtCore.QSize(0, 50))
        self.frame_4.setFrameShape(QtGui.QFrame.StyledPanel)
        self.frame_4.setFrameShadow(QtGui.QFrame.Raised)
        self.frame_4.setObjectName(_fromUtf8("frame_4"))
        self.verticalLayout_5 = QtGui.QVBoxLayout(self.frame_4)
        self.verticalLayout_5.setObjectName(_fromUtf8("verticalLayout_5"))
        self.label_4 = QtGui.QLabel(self.frame_4)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_4.sizePolicy().hasHeightForWidth())
        self.label_4.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_4.setFont(font)
        self.label_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_4.setObjectName(_fromUtf8("label_4"))
        self.verticalLayout_5.addWidget(self.label_4)
        self.line_2 = QtGui.QFrame(self.frame_4)
        self.line_2.setFrameShape(QtGui.QFrame.HLine)
        self.line_2.setFrameShadow(QtGui.QFrame.Sunken)
        self.line_2.setObjectName(_fromUtf8("line_2"))
        self.verticalLayout_5.addWidget(self.line_2)
        self.horizontalLayout_6 = QtGui.QHBoxLayout()
        self.horizontalLayout_6.setObjectName(_fromUtf8("horizontalLayout_6"))
        self.label_8 = QtGui.QLabel(self.frame_4)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_8.sizePolicy().hasHeightForWidth())
        self.label_8.setSizePolicy(sizePolicy)
        self.label_8.setAlignment(QtCore.Qt.AlignCenter)
        self.label_8.setObjectName(_fromUtf8("label_8"))
        self.horizontalLayout_6.addWidget(self.label_8)
        self.healthy_mean_SB = QtGui.QSpinBox(self.frame_4)
        self.healthy_mean_SB.setMaximum(256)
        self.healthy_mean_SB.setObjectName(_fromUtf8("healthy_mean_SB"))
        self.horizontalLayout_6.addWidget(self.healthy_mean_SB)
        self.verticalLayout_5.addLayout(self.horizontalLayout_6)
        self.horizontalLayout_7 = QtGui.QHBoxLayout()
        self.horizontalLayout_7.setObjectName(_fromUtf8("horizontalLayout_7"))
        self.label_9 = QtGui.QLabel(self.frame_4)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_9.sizePolicy().hasHeightForWidth())
        self.label_9.setSizePolicy(sizePolicy)
        self.label_9.setAlignment(QtCore.Qt.AlignCenter)
        self.label_9.setObjectName(_fromUtf8("label_9"))
        self.horizontalLayout_7.addWidget(self.label_9)
        self.healthy_std_SB = QtGui.QSpinBox(self.frame_4)
        self.healthy_std_SB.setObjectName(_fromUtf8("healthy_std_SB"))
        self.horizontalLayout_7.addWidget(self.healthy_std_SB)
        self.verticalLayout_5.addLayout(self.horizontalLayout_7)
        self.horizontalLayout_4.addWidget(self.frame_4)
        self.frame_5 = QtGui.QFrame(self.frame)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_5.sizePolicy().hasHeightForWidth())
        self.frame_5.setSizePolicy(sizePolicy)
        self.frame_5.setMinimumSize(QtCore.QSize(0, 50))
        self.frame_5.setFrameShape(QtGui.QFrame.StyledPanel)
        self.frame_5.setFrameShadow(QtGui.QFrame.Raised)
        self.frame_5.setObjectName(_fromUtf8("frame_5"))
        self.verticalLayout_6 = QtGui.QVBoxLayout(self.frame_5)
        self.verticalLayout_6.setObjectName(_fromUtf8("verticalLayout_6"))
        self.label_5 = QtGui.QLabel(self.frame_5)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_5.sizePolicy().hasHeightForWidth())
        self.label_5.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.label_5.setFont(font)
        self.label_5.setAlignment(QtCore.Qt.AlignCenter)
        self.label_5.setObjectName(_fromUtf8("label_5"))
        self.verticalLayout_6.addWidget(self.label_5)
        self.line_3 = QtGui.QFrame(self.frame_5)
        self.line_3.setFrameShape(QtGui.QFrame.HLine)
        self.line_3.setFrameShadow(QtGui.QFrame.Sunken)
        self.line_3.setObjectName(_fromUtf8("line_3"))
        self.verticalLayout_6.addWidget(self.line_3)
        self.horizontalLayout_8 = QtGui.QHBoxLayout()
        self.horizontalLayout_8.setObjectName(_fromUtf8("horizontalLayout_8"))
        self.label_10 = QtGui.QLabel(self.frame_5)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_10.sizePolicy().hasHeightForWidth())
        self.label_10.setSizePolicy(sizePolicy)
        self.label_10.setAlignment(QtCore.Qt.AlignCenter)
        self.label_10.setObjectName(_fromUtf8("label_10"))
        self.horizontalLayout_8.addWidget(self.label_10)
        self.hyper_mean_SB = QtGui.QSpinBox(self.frame_5)
        self.hyper_mean_SB.setMaximum(256)
        self.hyper_mean_SB.setObjectName(_fromUtf8("hyper_mean_SB"))
        self.horizontalLayout_8.addWidget(self.hyper_mean_SB)
        self.verticalLayout_6.addLayout(self.horizontalLayout_8)
        self.horizontalLayout_9 = QtGui.QHBoxLayout()
        self.horizontalLayout_9.setObjectName(_fromUtf8("horizontalLayout_9"))
        self.label_11 = QtGui.QLabel(self.frame_5)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_11.sizePolicy().hasHeightForWidth())
        self.label_11.setSizePolicy(sizePolicy)
        self.label_11.setAlignment(QtCore.Qt.AlignCenter)
        self.label_11.setObjectName(_fromUtf8("label_11"))
        self.horizontalLayout_9.addWidget(self.label_11)
        self.hyper_std_SB = QtGui.QSpinBox(self.frame_5)
        self.hyper_std_SB.setObjectName(_fromUtf8("hyper_std_SB"))
        self.horizontalLayout_9.addWidget(self.hyper_std_SB)
        self.verticalLayout_6.addLayout(self.horizontalLayout_9)
        self.horizontalLayout_4.addWidget(self.frame_5)
        self.verticalLayout.addLayout(self.horizontalLayout_4)
        self.horizontalLayout_10 = QtGui.QHBoxLayout()
        self.horizontalLayout_10.setObjectName(_fromUtf8("horizontalLayout_10"))
        self.apply_changes_BTN = QtGui.QPushButton(self.frame)
        self.apply_changes_BTN.setMaximumSize(QtCore.QSize(120, 16777215))
        self.apply_changes_BTN.setObjectName(_fromUtf8("apply_changes_BTN"))
        self.horizontalLayout_10.addWidget(self.apply_changes_BTN)
        self.verticalLayout.addLayout(self.horizontalLayout_10)
        self.frame_2 = QtGui.QFrame(self.splitter)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Maximum, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_2.sizePolicy().hasHeightForWidth())
        self.frame_2.setSizePolicy(sizePolicy)
        self.frame_2.setFrameShape(QtGui.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtGui.QFrame.Raised)
        self.frame_2.setObjectName(_fromUtf8("frame_2"))
        self.verticalLayout_2 = QtGui.QVBoxLayout(self.frame_2)
        self.verticalLayout_2.setObjectName(_fromUtf8("verticalLayout_2"))
        self.label_2 = QtGui.QLabel(self.frame_2)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_2.sizePolicy().hasHeightForWidth())
        self.label_2.setSizePolicy(sizePolicy)
        self.label_2.setMaximumSize(QtCore.QSize(16777215, 30))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setFrameShape(QtGui.QFrame.WinPanel)
        self.label_2.setFrameShadow(QtGui.QFrame.Plain)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName(_fromUtf8("label_2"))
        self.verticalLayout_2.addWidget(self.label_2)
        self.horizontalLayout_3 = QtGui.QHBoxLayout()
        self.horizontalLayout_3.setObjectName(_fromUtf8("horizontalLayout_3"))
        self.left_view_btns_F = QtGui.QFrame(self.frame_2)
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
        self.view_1_BTN = QtGui.QPushButton(self.left_view_btns_F)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.view_1_BTN.sizePolicy().hasHeightForWidth())
        self.view_1_BTN.setSizePolicy(sizePolicy)
        self.view_1_BTN.setMinimumSize(QtCore.QSize(40, 0))
        self.view_1_BTN.setText(_fromUtf8(""))
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(_fromUtf8("icons/Eye.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.view_1_BTN.setIcon(icon)
        self.view_1_BTN.setIconSize(QtCore.QSize(24, 24))
        self.view_1_BTN.setObjectName(_fromUtf8("view_1_BTN"))
        self.horizontalLayout_12.addWidget(self.view_1_BTN)
        self.show_im_1_BTN = QtGui.QPushButton(self.left_view_btns_F)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.show_im_1_BTN.sizePolicy().hasHeightForWidth())
        self.show_im_1_BTN.setSizePolicy(sizePolicy)
        self.show_im_1_BTN.setMinimumSize(QtCore.QSize(40, 0))
        self.show_im_1_BTN.setText(_fromUtf8(""))
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(_fromUtf8("icons/Stock graph.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.show_im_1_BTN.setIcon(icon1)
        self.show_im_1_BTN.setIconSize(QtCore.QSize(24, 24))
        self.show_im_1_BTN.setObjectName(_fromUtf8("show_im_1_BTN"))
        self.horizontalLayout_12.addWidget(self.show_im_1_BTN)
        self.show_labels_1_BTN = QtGui.QPushButton(self.left_view_btns_F)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.show_labels_1_BTN.sizePolicy().hasHeightForWidth())
        self.show_labels_1_BTN.setSizePolicy(sizePolicy)
        self.show_labels_1_BTN.setMinimumSize(QtCore.QSize(40, 0))
        self.show_labels_1_BTN.setText(_fromUtf8(""))
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap(_fromUtf8("icons/Blue tag.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.show_labels_1_BTN.setIcon(icon2)
        self.show_labels_1_BTN.setIconSize(QtCore.QSize(24, 24))
        self.show_labels_1_BTN.setObjectName(_fromUtf8("show_labels_1_BTN"))
        self.horizontalLayout_12.addWidget(self.show_labels_1_BTN)
        self.show_contours_1_BTN = QtGui.QPushButton(self.left_view_btns_F)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.show_contours_1_BTN.sizePolicy().hasHeightForWidth())
        self.show_contours_1_BTN.setSizePolicy(sizePolicy)
        self.show_contours_1_BTN.setMinimumSize(QtCore.QSize(40, 0))
        self.show_contours_1_BTN.setText(_fromUtf8(""))
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap(_fromUtf8("icons/Brush.png")), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.show_contours_1_BTN.setIcon(icon3)
        self.show_contours_1_BTN.setIconSize(QtCore.QSize(24, 24))
        self.show_contours_1_BTN.setObjectName(_fromUtf8("show_contours_1_BTN"))
        self.horizontalLayout_12.addWidget(self.show_contours_1_BTN)
        self.horizontalLayout_3.addWidget(self.left_view_btns_F)
        self.line_4 = QtGui.QFrame(self.frame_2)
        self.line_4.setFrameShape(QtGui.QFrame.VLine)
        self.line_4.setFrameShadow(QtGui.QFrame.Sunken)
        self.line_4.setObjectName(_fromUtf8("line_4"))
        self.horizontalLayout_3.addWidget(self.line_4)
        self.right_view_btns_F = QtGui.QFrame(self.frame_2)
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
        self.view_2_BTN = QtGui.QPushButton(self.right_view_btns_F)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.view_2_BTN.sizePolicy().hasHeightForWidth())
        self.view_2_BTN.setSizePolicy(sizePolicy)
        self.view_2_BTN.setMinimumSize(QtCore.QSize(40, 0))
        self.view_2_BTN.setText(_fromUtf8(""))
        self.view_2_BTN.setIcon(icon)
        self.view_2_BTN.setIconSize(QtCore.QSize(24, 24))
        self.view_2_BTN.setObjectName(_fromUtf8("view_2_BTN"))
        self.horizontalLayout_13.addWidget(self.view_2_BTN)
        self.show_im_2_BTN = QtGui.QPushButton(self.right_view_btns_F)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.show_im_2_BTN.sizePolicy().hasHeightForWidth())
        self.show_im_2_BTN.setSizePolicy(sizePolicy)
        self.show_im_2_BTN.setMinimumSize(QtCore.QSize(40, 0))
        self.show_im_2_BTN.setText(_fromUtf8(""))
        self.show_im_2_BTN.setIcon(icon1)
        self.show_im_2_BTN.setIconSize(QtCore.QSize(24, 24))
        self.show_im_2_BTN.setObjectName(_fromUtf8("show_im_2_BTN"))
        self.horizontalLayout_13.addWidget(self.show_im_2_BTN)
        self.show_labels_2_BTN = QtGui.QPushButton(self.right_view_btns_F)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.show_labels_2_BTN.sizePolicy().hasHeightForWidth())
        self.show_labels_2_BTN.setSizePolicy(sizePolicy)
        self.show_labels_2_BTN.setMinimumSize(QtCore.QSize(40, 0))
        self.show_labels_2_BTN.setText(_fromUtf8(""))
        self.show_labels_2_BTN.setIcon(icon2)
        self.show_labels_2_BTN.setIconSize(QtCore.QSize(24, 24))
        self.show_labels_2_BTN.setObjectName(_fromUtf8("show_labels_2_BTN"))
        self.horizontalLayout_13.addWidget(self.show_labels_2_BTN)
        self.show_contours_2_BTN = QtGui.QPushButton(self.right_view_btns_F)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.show_contours_2_BTN.sizePolicy().hasHeightForWidth())
        self.show_contours_2_BTN.setSizePolicy(sizePolicy)
        self.show_contours_2_BTN.setMinimumSize(QtCore.QSize(40, 0))
        self.show_contours_2_BTN.setText(_fromUtf8(""))
        self.show_contours_2_BTN.setIcon(icon3)
        self.show_contours_2_BTN.setIconSize(QtCore.QSize(24, 24))
        self.show_contours_2_BTN.setObjectName(_fromUtf8("show_contours_2_BTN"))
        self.horizontalLayout_13.addWidget(self.show_contours_2_BTN)
        self.horizontalLayout_3.addWidget(self.right_view_btns_F)
        self.verticalLayout_2.addLayout(self.horizontalLayout_3)
        self.viewer_WDT = QtGui.QWidget(self.frame_2)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.viewer_WDT.sizePolicy().hasHeightForWidth())
        self.viewer_WDT.setSizePolicy(sizePolicy)
        self.viewer_WDT.setMinimumSize(QtCore.QSize(0, 0))
        self.viewer_WDT.setObjectName(_fromUtf8("viewer_WDT"))
        self.verticalLayout_2.addWidget(self.viewer_WDT)
        self.horizontalLayout_11 = QtGui.QHBoxLayout()
        self.horizontalLayout_11.setObjectName(_fromUtf8("horizontalLayout_11"))
        self.slice_number_LBL = QtGui.QLabel(self.frame_2)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.slice_number_LBL.sizePolicy().hasHeightForWidth())
        self.slice_number_LBL.setSizePolicy(sizePolicy)
        self.slice_number_LBL.setMinimumSize(QtCore.QSize(100, 0))
        self.slice_number_LBL.setObjectName(_fromUtf8("slice_number_LBL"))
        self.horizontalLayout_11.addWidget(self.slice_number_LBL)
        self.horizontalScrollBar = QtGui.QScrollBar(self.frame_2)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.horizontalScrollBar.sizePolicy().hasHeightForWidth())
        self.horizontalScrollBar.setSizePolicy(sizePolicy)
        self.horizontalScrollBar.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalScrollBar.setObjectName(_fromUtf8("horizontalScrollBar"))
        self.horizontalLayout_11.addWidget(self.horizontalScrollBar)
        self.verticalLayout_2.addLayout(self.horizontalLayout_11)
        self.verticalLayout_4.addWidget(self.splitter)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1078, 25))
        self.menubar.setObjectName(_fromUtf8("menubar"))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtGui.QStatusBar(MainWindow)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow", None))
        self.label.setText(_translate("MainWindow", "COLOR MODEL", None))
        self.auto_colormodel_BTN.setText(_translate("MainWindow", "Automatic", None))
        self.label_3.setText(_translate("MainWindow", "hypodense", None))
        self.label_6.setText(_translate("MainWindow", "mean =", None))
        self.label_7.setText(_translate("MainWindow", "std =", None))
        self.label_4.setText(_translate("MainWindow", "healthy", None))
        self.label_8.setText(_translate("MainWindow", "mean =", None))
        self.label_9.setText(_translate("MainWindow", "std =", None))
        self.label_5.setText(_translate("MainWindow", "hyperdense", None))
        self.label_10.setText(_translate("MainWindow", "mean =", None))
        self.label_11.setText(_translate("MainWindow", "std =", None))
        self.apply_changes_BTN.setText(_translate("MainWindow", "Apply changes", None))
        self.label_2.setText(_translate("MainWindow", "VIEWER", None))
        self.slice_number_LBL.setText(_translate("MainWindow", "slice # =", None))

