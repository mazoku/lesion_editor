from PyQt4.QtCore import *
from PyQt4.QtGui import *
import sys
import numpy as np


class MyTableModel(QAbstractTableModel):
    def __init__(self, objects, data, headerdata=None, parent=None, *args):
        QAbstractTableModel.__init__(self, parent, *args)

        if headerdata is None:
            headerdata = ['label', 'area', 'mean density', 'compactness']

        self.objects = objects  # lesion list
        self.headerdata = headerdata
        self.data_arr = data  # data -> must be here to be able to del an object

    def rowCount(self, parent=QModelIndex()):
        # Number of rows corresponds to the number of objects
        return len(self.objects)

    def columnCount(self, parent=QModelIndex()):
        # Number of columns corresponds to the number of features
        return len(self.headerdata)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return QVariant()
        elif role != Qt.DisplayRole:
            return QVariant()
        data = (self.objects[index.row()].label, self.objects[index.row()].area, self.objects[index.row()].mean_density,
                self.objects[index.row()].compactness )
        if data[index.column()] is not None:
            if isinstance(data[index.column()], int):
                out = QVariant('%i' % data[index.column()])
            elif isinstance(data[index.column()], float):
                out = QVariant('%.3f' % data[index.column()])
            else:
                out = QVariant('x')
        else:
            out = QVariant(' ')
        return out

    def headerData(self, idx, orientation, role):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return QVariant(self.headerdata[idx])
        if orientation == Qt.Vertical and role == Qt.DisplayRole:
            return QVariant(idx)
        return QVariant()

    def removeRows(self, row, count=1, parent=QModelIndex()):
        self.beginRemoveRows(QModelIndex(), row, row + count - 1)
        # self.objects = self.objects[:row] + self.objects[row + count:]
        lbl = self.objects[row].label
        self.data_arr.labels = np.where(self.data_arr.labels == lbl, 0, self.data_arr.labels)
        self.objects.pop(row)
        self.endRemoveRows()
        print 'removed label', lbl
        # print self.labels
        return True


class MyWindow(QWidget):
    def __init__(self, objects, labels, header=None, *args):
        QWidget.__init__(self, *args)

        self.data_arr = objects
        self.labels = labels
        self.tablemodel = MyTableModel(self.data_arr, self.labels, header, self)
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
        print selected
        print deselected
        print '--'
        indexes = self.tableview.selectionModel().selectedRows()
        sel_mod = self.tableview.selectionModel()
        print self.tableview.selectionBehavior()
        pass
    #     indexes = self.tableview.selectionModel().selectedRows()
    #     for index in indexes:
    #         print index.row()
            # self.tableview.selectRow(index.row())

    def add_row(self):
        # try:
        #     new_idx = self.data[:, 0].max() + 1
        #     self.data = np.vstack((self.data, np.array([new_idx, 0, 0])))
        # except ValueError:
        #     self.data = np.array([[1, 0, 0]])
        # self.tablemodel.data = self.data
        # self.tableview.model().layoutChanged.emit()
        print 'row added'

    def del_row(self):
        # self.data = self.data[:-1, :]
        # self.tablemodel.data = self.data
        # self.tableview.model().layoutChanged.emit()
        indexes = self.tableview.selectionModel().selectedRows()
        for i in reversed(indexes):
            self.tablemodel.removeRow(i.row())
            # print 'deleted row', i.row()


if __name__ == "__main__":
    import Lesion

    class Data(object):
        def __init__(self, labels):
            self.labels = labels

    labels = np.array([[1, 1, 0, 2, 0],
                       [1, 1, 0, 2, 0],
                       [0, 0, 0, 2, 0],
                       [3, 0, 4, 0, 5],
                       [3, 0, 4, 0, 0]], dtype=np.int)
    labels = np.dstack((labels, labels, labels))

    data = Data(labels)

    lesions = Lesion.extract_lesions(labels, data=data.labels)

    app = QApplication(sys.argv)
    w = MyWindow(lesions, data)
    w.show()
    sys.exit(app.exec_())