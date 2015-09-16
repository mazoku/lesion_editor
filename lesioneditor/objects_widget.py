__author__ = 'tomas'

import sys
from PyQt4 import QtGui, QtCore
import numpy as np
import My_table_model as mtm

from qrangeslider import QRangeSlider

from objects_widget_GUI import Ui_Form

class Objects_widget(QtGui.QWidget):
    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self, parent)
        self.ui = Ui_Form()
        self.ui.setupUi(self)

        # area range slider
        self.area_RS = QRangeSlider()
        self.area_RS.setMin(0)
        self.area_RS.setMax(10000)
        self.area_RS.setRange(10, self.area_RS.max())
        self.ui.area_layout.addWidget(self.area_RS)

        # density range slider
        self.density_RS = QRangeSlider()
        self.density_RS.setMin(-50)
        self.density_RS.setMax(1000)
        self.density_RS.setRange(10, self.density_RS.max())
        self.ui.density_layout.addWidget(self.density_RS)

    def set_area_range(self, areas):
        self.area_RS.setMin(min(areas))
        self.area_RS.setMax(max(areas))
        self.area_RS.setRange()
        # self.ui.min_area_SL.setMaximum(max(areas))
        # self.ui.min_area_SL.setMinimum(min(areas))
        # self.ui.max_area_SL.setMaximum(max(areas))
        # self.ui.max_area_SL.setMinimum(min(areas))
        # self.ui.max_area_SL.setValue(max(areas))

    def keyPressEvent(self, QKeyEvent):
        print 'hist widget key event: ',
        if QKeyEvent.key() == QtCore.Qt.Key_Escape:
            print 'Escape'
            self.close()

if __name__ == '__main__':

    from objects_widget_GUI import Ui_Form
    app = QtGui.QApplication(sys.argv)

    objects_w = Objects_widget()
    objects_w.show()

    sys.exit(app.exec_())

# TODO: kdyz jsou handly blizko u sebe, neni videt cislo udavajici hodnotu -> nezobrazovat cisla, ale pridat line edity?