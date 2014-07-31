import sys

from PyQt4 import QtGui, QtCore
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar

import matplotlib.pyplot as plt

import numpy as np

class TumorVisualiser(QtGui.QMainWindow):

    def __init__(self):
        super(TumorVisualiser, self).__init__()

        self.init_UI_win()

    def init_UI_win(self):
        self.form_widget = FormWidget(self)

        self.setCentralWidget(self.form_widget)

class FormWidget(QtGui.QWidget):

    def __init__(self, window):

        self.win = window  # link to the main window

        super(FormWidget, self).__init__()
        self.init_UI_form()

    def init_UI_form(self):
        self.figure = plt.figure()
        # self.axes = self.figure.add_axes([0, 0, 1, 1])

        self.canvas = FigureCanvas(self.figure)

        self.slider = QtGui.QSlider(QtCore.Qt.Horizontal, self)
        slider_layout = QtGui.QVBoxLayout()
        slider_layout.addWidget(self.slider)
        slider_frame = QtGui.QFrame()
        slider_frame.setLayout(slider_layout)

        layout = QtGui.QVBoxLayout()
        layout.addWidget(self.canvas, 1)
        # layout.addWidget(self.slider)
        layout.addWidget(slider_frame)
        self.setLayout(layout)

        plt.subplot(121)
        plt.imshow(np.eye(60))

        plt.subplot(122)
        plt.imshow(np.eye(60)[::-1,:])

        self.canvas.draw()

if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    tv = TumorVisualiser()
    tv.show()
    sys.exit(app.exec_())
