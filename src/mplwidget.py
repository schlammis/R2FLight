
from PyQt5.QtWidgets import QWidget,QSizePolicy,QVBoxLayout,\
                        QHBoxLayout,QLabel
 
import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg,\
    NavigationToolbar2QT
from matplotlib import ticker

from matplotlib.figure import Figure
#from matplotlib.pyplot import subplots

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
            #self.label = QLabel('k2 viewer v0.1')
            self.hbl.addWidget(self.ntb)
        #self.hbl.addWidget(self.label)
        self.vbl.addWidget(self.canvas)
        self.vbl.addLayout(self.hbl)
        self.setLayout(self.vbl)
    
    def setfmt(self,xfmt,yfmt):
        self.canvas.setfmt(xfmt,yfmt)
    


class MplCanvas2(FigureCanvasQTAgg):
    def __init__(self):
        self.fig = Figure()
        ll=0.22
        bb=0.15
        h=0.35
        w=0.75
        dy=0.1
        self.ax1 = self.fig.add_axes((ll,bb+1*h+dy,w,h))
        self.ax2 = self.fig.add_axes((ll,bb+0*h,w,h))
        self.ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.5e"))
        self.ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.5e"))
        #self.ax1.ticklabel_format(useOffset=False)
        #self.ax2.ticklabel_format(useOffset=False)

        FigureCanvasQTAgg.__init__(self, self.fig)
        FigureCanvasQTAgg.setSizePolicy(self,
                                    QSizePolicy.Expanding,
                                    QSizePolicy.Expanding)
        FigureCanvasQTAgg.updateGeometry(self)
        
    def draw(self):   
        self.ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.5f"))
        self.ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.5f"))
        super(FigureCanvasQTAgg, self).draw()



    def setsamexscale(self):
        be1,en1 = self.ax1.get_xlim()
        be2,en2 = self.ax2.get_xlim()
        be = min(be1,be2)
        en = max(en1,en2)
        self.ax1.set_xlim(be,en)
        self.ax2.set_xlim(be,en)

        
        

class MplWidget2(QWidget):
    def __init__(self, parent = None):
        QWidget.__init__(self, parent)
        self.canvas = MplCanvas2()
        self.vbl = QVBoxLayout()
        self.hbl = QHBoxLayout()
        self.ntb = NavigationToolbar2QT(self.canvas,parent)
        self.label = QLabel('AFAirCap')
        self.hbl.addWidget(self.ntb)
        self.hbl.addWidget(self.label)
        self.vbl.addWidget(self.canvas)
        self.vbl.addLayout(self.hbl)
        self.setLayout(self.vbl)



class MplCanvas4(FigureCanvasQTAgg):
    def __init__(self):
        self.fig = Figure(figsize=(8, 8))
        ll=0.15
        bb=0.15
        h=0.2
        w=0.8
        ll=0.15
        bb=0.15
        h=0.35
        w=0.35
        dy=0.1
        dx=0.1
        self.ax1 = self.fig.add_axes((ll,bb+1*h+dx,w,h))
        self.ax2 = self.fig.add_axes((ll,bb+0*h,w,h))
        
        self.ax3 = self.fig.add_axes((ll+w+dy,bb+1*h+dx,w,h))
        self.ax4 = self.fig.add_axes((ll+w+dy,bb+0*h,w,h))
        
        
        FigureCanvasQTAgg.__init__(self, self.fig)
        FigureCanvasQTAgg.setSizePolicy(self,
                                    QSizePolicy.Expanding,
                                    QSizePolicy.Expanding)
        FigureCanvasQTAgg.updateGeometry(self)

class MplWidget4(QWidget):
    def __init__(self, parent = None):
        QWidget.__init__(self, parent)
        self.canvas = MplCanvas4()
        self.vbl = QVBoxLayout()
        self.hbl = QHBoxLayout()
        self.ntb = NavigationToolbar2QT(self.canvas,parent)
        self.label = QLabel('k2 viewer v0.1')
        self.hbl.addWidget(self.ntb)
        self.hbl.addWidget(self.label)
        self.vbl.addWidget(self.canvas)
        self.vbl.addLayout(self.hbl)
        self.setLayout(self.vbl)
