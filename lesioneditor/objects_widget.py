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

        # self.area_RS.endValueChanged.connect(self.max_area_changed_callback)
        # self.area_RS.startValueChanged.connect(self.min_area_changed_callback)

        # density range slider
        self.density_RS = QRangeSlider()
        self.density_RS.setMin(-50)
        self.density_RS.setMax(1000)
        self.density_RS.setRange(10, self.density_RS.max())
        self.ui.density_layout.addWidget(self.density_RS)

    # def max_area_changed_callback(self, value):
    #     print 'max area changed to ', value
    #
    # def min_area_changed(self, value):
    #     print 'min area changed to ', value

    def set_area_range(self, areas):
        self.area_RS.setMin(min(areas))
        self.area_RS.setMax(max(areas))
        self.area_RS.setRange()

    def set_density_range(self, densities):
        self.density_RS.setMin(min(densities))
        self.density_RS.setMax(max(densities))
        self.density_RS.setRange()

    def keyPressEvent(self, QKeyEvent):
        print 'hist widget key event: ',
        if QKeyEvent.key() == QtCore.Qt.Key_Escape:
            print 'Escape'
            if self.ui.objects_TV.selectedIndexes():
                self.ui.objects_TV.clearSelection()
            else:
                self.close()

    # self.cc.objects_filtration(self.selected_objects_labels, min_area=self.ui.min_area_SL.value(), max_area=value)
    #     # self.fill_table(self.cc.labels[self.cc.filtered_idxs], self.cc.areas[self.cc.filtered_idxs], self.cc.comps[self.cc.filtered_idxs])
    #     self.fill_table(self.cc.actual_data.lesions, self.cc.actual_data.labels, self.cc.filtered_idxs)
    #     # TODO: nasleduje prasarna
    #     if self.view_L.show_mode == self.view_L.SHOW_LABELS:
    #         self.show_labels_L_callback()
    #     if self.view_R.show_mode == self.view_R.SHOW_LABELS:
    #         self.show_labels_R_callback()

if __name__ == '__main__':

    from objects_widget_GUI import Ui_Form

    import Lesion
    app = QtGui.QApplication(sys.argv)

    labels = np.array([[1, 1, 0, 2, 0],
                       [1, 1, 0, 2, 0],
                       [0, 0, 0, 2, 0],
                       [3, 0, 4, 0, 5],
                       [3, 0, 4, 0, 0]], dtype=np.int)
    labels = np.dstack((labels, labels, labels))
    lesions = Lesion.extract_lesions(labels, data=labels)

    objects_w = Objects_widget()
    table_model = mtm.MyTableModel(lesions, labels)
    objects_w.ui.objects_TV.setModel(table_model)
    objects_w.show()

    sys.exit(app.exec_())