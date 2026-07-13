
from PyQt5.QtWidgets import QWidget,QSizePolicy,QVBoxLayout,\
                        QHBoxLayout

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg,\
    NavigationToolbar2QT
from matplotlib import ticker

from matplotlib.figure import Figure

import matplotlib.pyplot as plt

plt.rcParams['axes.labelsize'] = 8  # Font size for x and y labels
plt.rcParams['xtick.labelsize'] = 6  # Font size for x-axis tick labels
plt.rcParams['ytick.labelsize'] = 6  # Font size for y-axis tick labels


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self,rightax=False,inset=False):
        self.fig = Figure(figsize=(4, 4))
        self.ax1 = self.fig.add_subplot(111)
        self.ax1.set_box_aspect(1)
        if rightax:
            self.bx1 = self.ax1.twinx()
        elif inset:
            self.bx1 = self.fig.add_axes((0.35, 0.6, 0.15, 0.15))
        FigureCanvasQTAgg.__init__(self, self.fig)
        FigureCanvasQTAgg.setSizePolicy(self,
                                    QSizePolicy.Expanding,
                                    QSizePolicy.Expanding)
        FigureCanvasQTAgg.updateGeometry(self)
        self.yfmt = "%.5f"
        self.xfmt = "%.0f"

    def setfmt(self,xfmt,yfmt):
        self.xfmt = xfmt
        self.yfmt = yfmt
   
    def draw(self):
        self.ax1.xaxis.set_major_formatter(ticker.FormatStrFormatter(self.xfmt))
        self.ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter(self.yfmt))
        self.fig.tight_layout()
        super(FigureCanvasQTAgg, self).draw()


class MplWidget(QWidget):
    def __init__(self, parent = None,rightax=False,navigator=False,inset=False):
        QWidget.__init__(self, parent)
        self.canvas = MplCanvas(rightax,inset)
        self.vbl = QVBoxLayout()
        self.hbl = QHBoxLayout()
        if navigator:
            self.ntb = NavigationToolbar2QT(self.canvas,parent)
            self.hbl.addWidget(self.ntb)
        self.vbl.addWidget(self.canvas)
        self.vbl.addLayout(self.hbl)
        self.setLayout(self.vbl)
    
    def setfmt(self,xfmt,yfmt):
        self.canvas.setfmt(xfmt,yfmt)
