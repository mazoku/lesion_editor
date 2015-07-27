from PyQt4.QtCore import *
from PyQt4.QtGui import *
import sys
import numpy as np


class MyTableModel(QAbstractTableModel):
    def __init__(self, datain, headerdata=None, parent=None, *args):
        QAbstractTableModel.__init__(self, parent, *args)

        if headerdata is None:
            headerdata = ['label', 'area', 'compactness']

        self.data = datain
        self.headerdata = headerdata

    def rowCount(self, parent):
        # Number of rows corresponds to the number of objects
        return self.data.shape[0]

    def columnCount(self, parent):
        # Number of columns corresponds to the number of features
        return self.data.shape[1]

    def data(self, index, role):
        if not index.isValid():
            return QVariant()
        elif role != Qt.DisplayRole:
            return QVariant()
        return QVariant('%.3f'%self.data[index.row(), index.column()])

    def headerData(self, idx, orientation, role):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return QVariant(self.headerdata[idx])
        if orientation == Qt.Vertical and role == Qt.DisplayRole:
            return QVariant(idx)
        return QVariant()



class MyWindow(QWidget):
    def __init__(self, data, header=None, *args):
        QWidget.__init__(self, *args)

        self.data = data
        self.tablemodel = MyTableModel(self.data, header, self)
        self.tableview = QTableView()
        self.tableview.setModel(self.tablemodel)
        self.tableview.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.tableview.selectionModel().selectionChanged.connect(self.selection_changed)

        add_BTN = QPushButton('add')
        self.connect(add_BTN, SIGNAL('clicked()'), self.add_row)

        del_BTN = QPushButton('delete')
        self.connect(del_BTN, SIGNAL('clicked()'), self.del_row)

        layoutV = QVBoxLayout(self)
        layoutV.addWidget(self.tableview)

        layoutH = QHBoxLayout(self)
        layoutH.addWidget(add_BTN)
        layoutH.addWidget(del_BTN)

        btn_frame = QFrame()
        btn_frame.setLayout(layoutH)

        layoutV.addWidget(btn_frame)
        self.setLayout(layoutV)

    def selection_changed(self, selected, deselected):
        indexes = self.tableview.selectionModel().selectedRows()
        for index in indexes:
            print index.row()
            # self.tableview.selectRow(index.row())

    def add_row(self):
        try:
            new_idx = self.data[:, 0].max() + 1
            self.data = np.vstack((self.data, np.array([new_idx, 0, 0])))
        except ValueError:
            self.data = np.array([[1, 0, 0]])
        self.tablemodel.data = self.data
        self.tableview.model().layoutChanged.emit()
        print 'row added'

    def del_row(self):
        self.data = self.data[:-1, :]
        self.tablemodel.data = self.data
        self.tableview.model().layoutChanged.emit()
        print 'row deleted'


if __name__ == "__main__":

    # my_array = [['1', '00', '01'],
    #             ['2', '10', '11'],
    #             ['3', '20', '21']]

    areas = [10, 20, 30, 40]
    comps = [0.1, 0.2, 0.3, 0.4]
    idxs = np.arange(1, len(areas)+1)
    data = np.vstack((idxs, areas, comps)).T

    app = QApplication(sys.argv)
    w = MyWindow(data)
    w.show()
    sys.exit(app.exec_())