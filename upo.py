from PyQt4.QtCore import *
from PyQt4.QtGui import *
import sys


class MyWindow(QWidget):
    def __init__(self, header, my_array, *args):
        QWidget.__init__(self, *args)

        tablemodel = MyTableModel(header, my_array, self)
        tableview = QTableView()
        tableview.setModel(tablemodel)

        layout = QVBoxLayout(self)
        layout.addWidget(tableview)
        self.setLayout(layout)


class MyTableModel(QAbstractTableModel):
    def __init__(self, headerdata, datain, parent=None, *args):
        QAbstractTableModel.__init__(self, parent, *args)

        self.arraydata = datain
        self.headerdata = headerdata


    def rowCount(self, parent):
        return len(self.arraydata)


    def columnCount(self, parent):
        return len(self.arraydata[0])


    def data(self, index, role):
        if not index.isValid():
            return QVariant()
        elif role != Qt.DisplayRole:
            return QVariant()
        return QVariant(self.arraydata[index.row()][index.column()])


    def headerData(self, col, orientation, role):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return QVariant(self.headerdata[col])
        return QVariant()


if __name__ == "__main__":

    header = ['idx','areas', 'compactness']
    my_array = [['1', '00', '01'],
                ['2', '10', '11'],
                ['3', '20', '21']]

    app = QApplication(sys.argv)
    w = MyWindow(header, my_array)
    w.show()
    sys.exit(app.exec_())