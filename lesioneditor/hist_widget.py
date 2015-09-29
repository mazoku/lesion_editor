__author__ = 'tomas'

import sys
from PyQt4 import QtGui, QtCore
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import numpy as np
import skimage.exposure as skiexp
import scipy.stats as scista

from hist_widget_GUI import Ui_Form

# Main widget containing figures etc
class Hist_widget(QtGui.QWidget):

    heal_parameter_changed = QtCore.pyqtSignal(int, int, basestring)
    hypo_parameter_changed = QtCore.pyqtSignal(int, int, basestring)
    hyper_parameter_changed = QtCore.pyqtSignal(int, int, basestring)

    def __init__(self, data=None, mask=None, models=None, hist=None, bins=None, parent=None, unaries_as_cdf=False, params=None):
        QtGui.QWidget.__init__(self, parent)
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        self.setup_ui()

        self.params = params
        self.hist = hist
        self.bins = bins
        # if self.hist is None and self.bins is None:
        #     if self.data is not None:
        #         # print 'No data given - returning.'
        #         # return
        #         self.hist, self.bins = skiexp.histogram(self.data, nbins=256)

        # self.models = models  # color models
        #
        # if self.models is not None:
        #     self.rv_heal = models['heal']
        #     self.rv_hypo = models['hypo']
        #     self.rv_hyper = models['hyper']
        #     self.setup_ui()
        # else:
        #     self.rv_heal = None
        #     self.rv_hypo = None
        #     self.rv_hyper = None

        self.rv_heal = scista.norm(0, 1)
        self.rv_hypo = scista.norm(0, 1)
        self.rv_hyper = scista.norm(0, 1)

        self.unaries_as_cdf = unaries_as_cdf

        self.figure = plt.figure()
        self.axes = self.figure.add_axes([0.08, 0.05, 0.91, 0.9])
        self.canvas = FigureCanvas(self.figure)

        layout = QtGui.QVBoxLayout()
        layout.addWidget(self.canvas, 1)
        self.ui.histogram_F.setLayout(layout)

        if data is not None:
            self.set_data(data)#, mask)
        if models is not None:
            self.set_models(models)
        else:
            self.models = None

        # seting up min and max values

        # if self.params and self.params.has_key('data_min'):
        #     self.data_min = self.params['data_min']
        # elif self.data is not None:
        #     self.data_min = self.data.min()
        # else:
        #     self.data_min = 0
        # if self.params and self.params.has_key('datam_max'):
        #     self.data_max = self.params['data_max']
        # elif self.data is not None:
        #     self.data_max = self.data.max()
        # else:
        #     self.data_max = 0
        #
        # self.ui.hypo_mean_SL.setMinimum(self.data_min)
        # self.ui.heal_mean_SL.setMinimum(self.data_min)
        # self.ui.hyper_mean_SL.setMinimum(self.data_min)
        #
        # self.ui.hypo_mean_SL.setMaximum(self.data_max)
        # self.ui.heal_mean_SL.setMaximum(self.data_max)
        # self.ui.hyper_mean_SL.setMaximum(self.data_max)
        #
        # self.update_figures()

    def set_data(self, data):#, mask):
        self.data = data
        # self.mask = mask
        # if self.data is not None:
        self.hist, self.bins = skiexp.histogram(self.data, nbins=1000)
            # if mask is not None:
            #     self.data_m = self.data[np.nonzero(self.mask)]
            # else:
            #     self.data_m = self.data

        if self.params and self.params.has_key('data_min'):
            self.data_min = self.params['data_min']
        # elif self.data is not None:
        self.data_min = self.data.min()
        # else:
        #     self.data_min = 0
        if self.params and self.params.has_key('datam_max'):
            self.data_max = self.params['data_max']
        # elif self.data is not None:
        self.data_max = self.data.max()
        # else:
        #     self.data_max = 0

        self.ui.hypo_mean_SL.setMinimum(self.data_min)
        self.ui.heal_mean_SL.setMinimum(self.data_min)
        self.ui.hyper_mean_SL.setMinimum(self.data_min)

        self.ui.hypo_mean_SL.setMaximum(self.data_max)
        self.ui.heal_mean_SL.setMaximum(self.data_max)
        self.ui.hyper_mean_SL.setMaximum(self.data_max)

        self.update_figures()

    def set_models(self, models):
        self.models = models  # color models

        if self.models is not None:
            self.rv_heal = models['heal']
            self.rv_hypo = models['hypo']
            self.rv_hyper = models['hyper']
            # self.setup_ui()
            self.setup_ranges()

            self.update_heal_rv(self.rv_heal)
            self.update_hypo_rv(self.rv_hypo)
            self.update_hyper_rv(self.rv_hyper)
            self.update_figures()
        else:
            self.rv_heal = None
            self.rv_hypo = None
            self.rv_hyper = None

    def setup_ranges(self):
        # filling line edits and spinboxes
        self.ui.hypo_mean_LE.setText('%i'%int(self.rv_hypo.mean()))
        self.ui.hypo_mean_SL.setValue(int(self.rv_hypo.mean()))
        self.ui.hypo_std_SB.setValue(self.rv_hypo.std())

        self.ui.heal_mean_LE.setText('%i'%int(self.rv_heal.mean()))
        self.ui.heal_mean_SL.setValue(int(self.rv_heal.mean()))
        self.ui.heal_std_SB.setValue(self.rv_heal.std())

        self.ui.hyper_mean_LE.setText('%i'%int(self.rv_hyper.mean()))
        self.ui.hyper_mean_SL.setValue(int(self.rv_hyper.mean()))
        self.ui.hyper_std_SB.setValue(self.rv_hyper.std())

    def setup_ui(self):
        # self.setup_ranges()

        # callbacks - hypodense mean
        self.ui.hypo_mean_SL.valueChanged.connect(self.hypo_mean_SL_callback)
        hypo_mean_val = QtGui.QIntValidator(self.ui.hypo_mean_SL.minimum(), self.ui.hypo_mean_SL.maximum())
        self.ui.hypo_mean_LE.setValidator(hypo_mean_val)
        self.ui.hypo_mean_LE.textChanged.connect(self.hypo_mean_LE_callback)

        # callbacks - heal mean
        self.ui.heal_mean_SL.valueChanged.connect(self.heal_mean_SL_callback)
        heal_mean_val = QtGui.QIntValidator(self.ui.heal_mean_SL.minimum(), self.ui.heal_mean_SL.maximum())
        self.ui.heal_mean_LE.setValidator(heal_mean_val)
        self.ui.heal_mean_LE.textChanged.connect(self.heal_mean_LE_callback)

        # callbacks - hyper mean
        self.ui.hyper_mean_SL.valueChanged.connect(self.hyper_mean_SL_callback)
        hyper_mean_val = QtGui.QIntValidator(self.ui.hyper_mean_SL.minimum(), self.ui.hyper_mean_SL.maximum())
        self.ui.hyper_mean_LE.setValidator(hyper_mean_val)
        self.ui.hyper_mean_LE.textChanged.connect(self.hyper_mean_LE_callback)

        # callbacks - stds
        self.ui.hypo_std_SB.valueChanged.connect(self.hypo_std_SB_callback)
        self.ui.hyper_std_SB.valueChanged.connect(self.hyper_std_SB_callback)
        self.ui.heal_std_SB.valueChanged.connect(self.heal_std_SB_callback)

    # def set_models(self, models):
    #     self.models = models
    #
    #     self.rv_heal = models['heal']
    #     self.rv_hypo = models['hypo']
    #     self.rv_hyper = models['hyper']
    #
    #     self.setup_ui()
    #
    #     self.update_figures()

    def hypo_mean_SL_callback(self, value):
        self.rv_hypo = scista.norm(value, self.rv_hypo.std())
        self.update_hypo_rv(self.rv_hypo)
        self.update_figures()

    def heal_mean_SL_callback(self, value):
        self.rv_heal = scista.norm(value, self.rv_heal.std())
        self.update_heal_rv(self.rv_heal)
        self.update_figures()

    def hyper_mean_SL_callback(self, value):
        self.rv_hyper = scista.norm(value, self.rv_hyper.std())
        self.update_hyper_rv(self.rv_hyper)
        self.update_figures()

    def hypo_mean_LE_callback(self, value):
        try:  # must be due to the possibility that no character could be entered
            self.ui.hypo_mean_SL.setValue(int(value))
        except:
            pass

    def heal_mean_LE_callback(self, value):
        try:  # must be due to the possibility that no character could be entered
            self.ui.heal_mean_SL.setValue(int(value))
        except:
            pass

    def hyper_mean_LE_callback(self, value):
        try:  # must be due to the possibility that no character could be entered
            self.ui.hyper_mean_SL.setValue(int(value))
        except:
            pass

    def hypo_std_SB_callback(self, value):
        # self.statusBar().showMessage('Hypodense model updated thru spin box.')
        self.rv_hypo = scista.norm(self.rv_hypo.mean(), value)
        self.update_hypo_rv(self.rv_hypo)
        self.update_figures()

    def hyper_std_SB_callback(self, value):
        # self.statusBar().showMessage('Hyperdense model updated thru spin box.')
        self.rv_hyper = scista.norm(self.rv_hyper.mean(), value)
        self.update_hyper_rv(self.rv_hyper)
        self.update_figures()

    def heal_std_SB_callback(self, value):
        # self.statusBar().showMessage('Healthy model updated thru spin box.')
        self.rv_heal = scista.norm(self.rv_heal.mean(), value)
        self.update_heal_rv(self.rv_heal)
        self.update_figures()

    def update_heal_rv(self, new_rv):
        self.rv_heal = new_rv
        if int(self.ui.heal_mean_LE.text()) != self.rv_heal.mean():
            self.ui.heal_mean_LE.setText('%i'%int(self.rv_heal.mean()))
        if self.ui.heal_std_SB.value() != self.rv_heal.std():
            self.ui.heal_std_SB.setValue(self.rv_heal.std())
        self.heal_parameter_changed.emit(new_rv.mean(), new_rv.std(), 'heal')

    def update_hypo_rv(self, new_rv):
        self.rv_hypo = new_rv
        if int(self.ui.hypo_mean_LE.text()) != self.rv_hypo.mean():
            self.ui.hypo_mean_LE.setText('%i'%int(self.rv_hypo.mean()))
        if self.ui.hypo_std_SB.value() != self.rv_hypo.std():
            self.ui.hypo_std_SB.setValue(self.rv_hypo.std())
        self.hypo_parameter_changed.emit(new_rv.mean(), new_rv.std(), 'hypo')

    def update_hyper_rv(self, new_rv):
        self.rv_hyper = new_rv
        if int(self.ui.hyper_mean_LE.text()) != self.rv_hyper.mean():
            self.ui.hyper_mean_LE.setText('%i'%int(self.rv_hyper.mean()))
        if self.ui.hyper_std_SB.value() != self.rv_hyper.std():
            self.ui.hyper_std_SB.setValue(self.rv_hyper.std())
        self.hyper_parameter_changed.emit(new_rv.mean(), new_rv.std(), 'hyper')

    def update_figures(self):
        plt.figure(self.figure.number)
        x = np.arange(self.data.min(), self.data.max())#, (self.data.max() - self.data.min()) / 100)  # artificial x-axis
        # self.figure.gca().cla()  # clearing the figure, just to be sure

        # plt.subplot(411)
        plt.plot(self.bins, self.hist, 'k')
        plt.hold(True)
        # if self.rv_heal is not None and self.rv_hypo is not None and self.rv_hyper is not None:
        if self.models is not None:
            healthy_y = self.rv_heal.pdf(x)
            if self.unaries_as_cdf:
                hypo_y = (1 - self.rv_hypo.cdf(x)) * self.rv_heal.pdf(self.rv_heal.mean())
                hyper_y = self.rv_hyper.cdf(x) * self.rv_heal.pdf(self.rv_heal.mean())
            else:
                hypo_y = self.rv_hypo.pdf(x)
                hyper_y = self.rv_hyper.pdf(x)
            y_max = max(healthy_y.max(), hypo_y.max(), hyper_y.max())
            fac = self.hist.max() / y_max

            plt.plot(x, fac * healthy_y, 'g', linewidth=2)
            plt.plot(x, fac * hypo_y, 'b', linewidth=2)
            plt.plot(x, fac * hyper_y, 'r', linewidth=2)
        ax = plt.axis()
        # plt.axis([0, 256, ax[2], ax[3]])
        plt.gca().tick_params(direction='in', pad=1)
        plt.hold(False)
        # plt.grid(True)

        self.canvas.draw()

    def keyPressEvent(self, QKeyEvent):
        print 'hist widget key event: ',
        if QKeyEvent.key() == QtCore.Qt.Key_Escape:
            print 'Escape'
            self.close()

if __name__ == '__main__':

    from hist_widget_GUI import Ui_Form
    import skimage.io as skiio
    import skimage.exposure as skiexp
    import scipy.stats as scista

    app = QtGui.QApplication(sys.argv)

    img = skiio.imread('/home/tomas/Dropbox/images/puma.png', as_grey=True)
    img = skiexp.rescale_intensity(img, in_range=(0, 1), out_range=np.uint8)

    models = dict()
    rv_hypo = scista.norm(50, 4)
    rv_heal = scista.norm(125, 10)
    rv_hyper = scista.norm(200, 5)
    models['hypo'] = rv_hypo
    models['heal'] = rv_heal
    models['hyper'] = rv_hyper
    params = {'data_min': img.min(), 'data_max':img.max()}
    hist_w = Hist_widget(data=img, params=params, unaries_as_cdf=True)
    hist_w.set_models(models)
    hist_w.show()

    sys.exit(app.exec_())