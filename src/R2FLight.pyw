import os,sys
import traceback
import ctypes
import datetime
import time

import mplwidget
import numpy as np
from matplotlib.colors import to_rgb
import spectral3
from Meas3 import Meas
import CustomData
import R2FConfig
import R2FLightAux
from TZA import TZA
import mystat

from PyQt5.QtCore import (
    QMutex,
    QThread,
    QTimer,
    pyqtSignal)

from PyQt5.QtWidgets import (
    QApplication,
    QAction,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QGridLayout,
    QWidget,
    QTabWidget,
    QSpacerItem,
    QSizePolicy,
    QStatusBar,
    QCheckBox,
    QDoubleSpinBox,
    QProgressBar,
    QRadioButton,
    QButtonGroup,
    QTextEdit,
    QLineEdit,
)


class MainWindow(QMainWindow):
    stopDVM = pyqtSignal()

    def __init__(self, mutex):
        super().__init__()
        self.quit = False
        self.loopfinished = False
        self._meas_running = False
        # True only while self.rData's fit actually corresponds to the cycle
        # currently being displayed -- cleared the moment a new cycle starts
        # collecting points, so stale fit overlays don't linger on screen.
        self._fit_is_current = False
        # live V2/V3/eta1/eta3 preview accumulated during current ellipse,
        # split by switch state, so the scatter tab looks the same whether a
        # measurement is in progress or complete
        self._partial_v2A = []
        self._partial_v2B = []
        self._partial_v3A = []
        self._partial_v3B = []
        self._partial_eta1A = []
        self._partial_eta1B = []
        self._partial_eta3A = []
        self._partial_eta3B = []
        self.mutex = mutex
        self.thread = QThread()
        self.cfg = R2FConfig.CFG()
        self.Npts = self.cfg.getintkey('NPTS')  # number of points in one full ellipse sweep (one switch position)
        self.fsig = self.cfg.getfloatkey('FSIG')
        self.mytext = []
        self.mytextmaxlen = 1000

        dummy_config = CustomData.NPointsConfig(1000, 800000, Nhars=self.cfg.getintkey('NHARS'))
        self.rData   = CustomData.NPoints(dummy_config)
        self.rSet    = CustomData.FourChannels(1000, 800000, 2, [], [], [], [], 0, 0, 0, ts=-1)
        self.allData = CustomData.DAQBuffer(1000, cols=6)

        self.t0 = time.time()
        tnow = time.localtime(self.t0)
        bd = os.path.join(r'c:\DATA\R2F', time.strftime('%Y%m', tnow))
        if not os.path.isdir(bd):
            os.mkdir(bd)
        bd = os.path.join(bd, time.strftime('%Y%m%d', tnow))
        if not os.path.isdir(bd):
            os.mkdir(bd)
        self.yyyymmdir = bd
        self.fn    = 'run' + time.strftime('%H%M%S', tnow) + '.dat'
        self.rawfn = 'raw' + time.strftime('%H%M%S', tnow) + '.dat'

        try:
            self.tza = TZA()
        except RuntimeError as e:
            self.tza = None
            print(f'TZA: {e}')

        self._setup_ui()
        self.myprint("Welcome to R2FNw")
        if self.tza is None:
            self.myprint("WARNING: TZA amplifier not found on any COM port")

        self.goagain()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _setup_ui(self):
        self.setWindowTitle('R2FNw')

        # --- plot grids ---
        self.rawplots = np.empty((2, 2), dtype=object)
        for i in range(2):
            for j in range(2):
                self.rawplots[i, j] = mplwidget.MplWidget(rightax=False)

        self.scatterplots = np.empty((2, 2), dtype=object)
        for i in range(2):
            for j in range(2):
                self.scatterplots[i, j] = mplwidget.MplWidget(rightax=False)
                self.scatterplots[i, j].setfmt("%.5f", "%.6f")

        self.residualplots = np.empty((2, 2), dtype=object)
        for i in range(2):
            for j in range(2):
                self.residualplots[i, j] = mplwidget.MplWidget(rightax=False)

        self.resultplots = np.empty((2, 2), dtype=object)
        for i in range(2):
            for j in range(2):
                self.resultplots[i, j] = mplwidget.MplWidget(rightax=False)
                self.resultplots[i, j].setfmt("%.0f", "%.6f")

        self.output = QTextEdit(self)
        self.output.resize(540, 200)
        self.output.setReadOnly(True)

        self.cfgWidget, self.cfgFields = self._build_config_widget()

        # --- menu bar ---
        file_menu = self.menuBar().addMenu('File')
        quit_action = QAction('Quit', self)
        quit_action.triggered.connect(self.finishUp)
        file_menu.addAction(quit_action)

        # --- side-panel controls ---
        vlayout = QVBoxLayout()
        vlayout.addItem(QSpacerItem(60, 20, QSizePolicy.Fixed, QSizePolicy.Fixed))

        self.buQuit = QPushButton("Quit")
        self.bufp   = QPushButton("f  +")
        self.bufm   = QPushButton("f  -")

        self.fstep = QDoubleSpinBox()
        self.fstep.setRange(0.001, 1000.0)
        self.fstep.setValue(1.0)
        self.fstep.setDecimals(3)
        self.fstep.setSuffix(' Hz')
        self.fstep.setPrefix('step: ')

        self.cbAutoFreq = QCheckBox("change frequency")
        self.cbAutoFreq.setChecked(bool(self.cfg.getintkey('AUTOFREQ')))
        self.cbModOff = QCheckBox("modulation off")
        self.cbModOff.setChecked(bool(self.cfg.getintkey('MODOFF')))

        self.buPause = QPushButton("Pause")
        self.buPause.setCheckable(True)

        vlayout.addWidget(self.buQuit)
        vlayout.addWidget(self.buPause)
        vlayout.addWidget(self.bufp)
        vlayout.addWidget(self.bufm)
        vlayout.addWidget(self.fstep)
        vlayout.addWidget(self.cbAutoFreq)
        vlayout.addWidget(self.cbModOff)
        vlayout.addWidget(QLabel('current f:'))
        self.laf = QLabel()
        self._update_freq_label()
        vlayout.addWidget(self.laf)
        vlayout.addItem(QSpacerItem(60, 10, QSizePolicy.Fixed, QSizePolicy.Fixed))

        # --- TZA gain radio buttons ---
        vlayout.addWidget(QLabel('TZA gain:'))
        self._tza_gain_group = QButtonGroup(self)
        DEFAULT_GAIN = self.cfg.getintkey('TZAGAIN')
        if not (1 <= DEFAULT_GAIN <= 6):
            DEFAULT_GAIN = 4
        _tza_labels = {1: '1 k\u03a9', 2: '10 k\u03a9', 3: '100 k\u03a9',
                       4: '1 M\u03a9', 5: '10 M\u03a9'}
        for g in range(1, 6):
            rb = QRadioButton(f'V{g}  ({_tza_labels[g]})')
            self._tza_gain_group.addButton(rb, g)
            vlayout.addWidget(rb)
            if g == DEFAULT_GAIN:
                rb.setChecked(True)
        self._tza_gain_group.idClicked.connect(self._on_tza_gain)
        # apply default gain to device now that UI exists
        if self.tza is not None:
            self.tza.set_gain(DEFAULT_GAIN)

        vlayout.addItem(QSpacerItem(60, 20, QSizePolicy.Fixed, QSizePolicy.Expanding))

        # --- main layout ---
        glayout = QHBoxLayout()
        glayout.addLayout(vlayout)
        self.tabWidget = MyTabWidget(self)
        glayout.addWidget(self.tabWidget)

        central_widget = QWidget()
        central_widget.setLayout(glayout)
        self.setCentralWidget(central_widget)

        # --- status bar ---
        self.progressBar = QProgressBar()
        self.progressBar.setMaximumWidth(400)
        self.progressBar.setRange(0, self.Npts * 2)
        self.progressBar.setValue(0)

        sb_widget = QWidget()
        sb_layout = QHBoxLayout(sb_widget)
        sb_layout.setContentsMargins(0, 0, 0, 0)
        sb_layout.addWidget(self.progressBar)

        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.addPermanentWidget(sb_widget)
        self.statusBar.showMessage('Welcome to R2FNw', 5000)

        # --- signals ---
        self.buQuit.clicked.connect(self.finishUp)
        self.buPause.toggled.connect(self._on_pause_toggled)
        self.bufp.clicked.connect(self.fp)
        self.bufm.clicked.connect(self.fm)
        self.cbAutoFreq.toggled.connect(self._on_autofreq_toggled)
        self.cbModOff.toggled.connect(self._on_modoff_toggled)

    def _update_freq_label(self):
        self.laf.setText(f'{self.fsig:8.5f} Hz')
        self.cfg.setfloatkey('FSIG', self.fsig)

    @staticmethod
    def _cfg_key_is_float(key):
        return isinstance(R2FConfig.CFG.std[key], float)

    def _cfg_get(self, key):
        return self.cfg.getfloatkey(key) if self._cfg_key_is_float(key) else self.cfg.getintkey(key)

    def _build_config_widget(self):
        """Builds an editable form for the keys in R2FLight.ini."""
        widget = QWidget()
        form = QFormLayout()
        fields = {}
        for key in sorted(R2FConfig.CFG.std):
            edit = QLineEdit(str(self._cfg_get(key)))
            form.addRow(key, edit)
            fields[key] = edit
        save_btn = QPushButton('Save')
        save_btn.clicked.connect(self._save_config)
        form.addRow(save_btn)
        widget.setLayout(form)
        return widget, fields

    def _refresh_config_fields(self):
        for key, edit in self.cfgFields.items():
            edit.setText(str(self._cfg_get(key)))

    def _save_config(self):
        for key, edit in self.cfgFields.items():
            is_float = self._cfg_key_is_float(key)
            try:
                val = float(edit.text()) if is_float else int(edit.text())
            except ValueError:
                kind = 'a number' if is_float else 'an integer'
                self.myprint(f'Config: "{key}" must be {kind}, not saved')
                continue
            if is_float:
                self.cfg.setfloatkey(key, val)
            else:
                self.cfg.setintkey(key, val)
            edit.setText(str(val))
        self.myprint('Config saved to R2FLight.ini')

    # ------------------------------------------------------------------
    # Measurement loop
    # ------------------------------------------------------------------

    def onNewData(self, MyData: CustomData.NPoints):
        self.rData = MyData
        self._fit_is_current = True
        line = [self.rData.Res["ts"], self.rData.Res["fsig"],
                np.real(self.rData.Res["mratio1"]), np.imag(self.rData.Res["mratio1"]),
                self.rData.Res["R"], self.rData.Res["fnew"]]

        if self.cbAutoFreq.isChecked():
            self.fsig = self.rData.Res["fnew"]
        self._update_freq_label()

        self.allData.add_point(line)

        with open(os.path.join(self.yyyymmdir, self.fn), "a", encoding="ascii") as file:
            oline = ''.join(f'{item:18.8f} ' for item in line) + '\n'
            file.write(oline)

        dat = np.vstack((
            np.ones(len(self.rData.ave4[:, 0])) * self.rData.Res["ts"],
            np.ones(len(self.rData.ave4[:, 0])) * self.rData.Res["fsig"],
            np.real(self.rData.ave4[:, 0]), np.imag(self.rData.ave4[:, 0]),
            np.real(self.rData.ave4[:, 1]), np.imag(self.rData.ave4[:, 1]),
            np.real(self.rData.ave4[:, 2]), np.imag(self.rData.ave4[:, 2]))).T

        with open(os.path.join(self.yyyymmdir, self.rawfn), "a", encoding="ascii") as file:
            for l in dat:
                s = f'{l[0]:10.0f} ' + ''.join(f'{c:18.8f} ' for c in l[1:]) + '\n'
                file.write(s)

        self.myprint('Data ready, now plotting')
        self.replot()
        self.statusBar.showMessage('Data ready, now plotting', 5000)

    def onNewSet(self, MySet: CustomData.FourChannels):
        self.rSet = MySet
        self.progressBar.setValue(MySet.i + 1)
        if MySet.i == 0:
            self._fit_is_current = False
            self._partial_v2A = []
            self._partial_v2B = []
            self._partial_v3A = []
            self._partial_v3B = []
            self._partial_eta1A = []
            self._partial_eta1B = []
            self._partial_eta3A = []
            self._partial_eta3B = []
        # live, per-point preview of V2/V3 (phase-derotated, matching raw8) and
        # eta1/eta3 (V2/V1, V3/V1) as points stream in, ahead of the final
        # switch-averaged/fit values -- split by switch state (first Npts
        # points are switch A, next Npts are switch B)
        Vc0 = MySet.Data[0].Vc
        if abs(Vc0) > 0:
            cf = np.exp(-1j * np.angle(Vc0))
            v2 = MySet.Data[1].Vc * cf
            v3 = MySet.Data[2].Vc * cf
            eta1 = MySet.Data[1].Vc / Vc0
            eta3 = MySet.Data[2].Vc / Vc0
            if MySet.i < self.Npts:
                self._partial_v2A.append(v2)
                self._partial_v3A.append(v3)
                self._partial_eta1A.append(eta1)
                self._partial_eta3A.append(eta3)
            else:
                self._partial_v2B.append(v2)
                self._partial_v3B.append(v3)
                self._partial_eta1B.append(eta1)
                self._partial_eta3B.append(eta3)
        self.replot()

    def _on_pause_toggled(self, paused):
        if paused:
            self.buPause.setText("Resume")
            self.statusBar.showMessage('Paused — will stop after current ellipse')
            self.myprint('Paused')
        else:
            self.buPause.setText("Pause")
            self.statusBar.showMessage('Resuming...')
            self.myprint('Resumed')
            # if goagain was held back while paused, restart now
            if not self._meas_running:
                self.goagain()

    def _on_autofreq_toggled(self, checked):
        self.cfg.setintkey('AUTOFREQ', int(checked))

    def _on_modoff_toggled(self, checked):
        self.cfg.setintkey('MODOFF', int(checked))

    def goagain(self):
        self._meas_running = False
        if self.buPause.isChecked():
            return   # hold here; _on_pause_toggled will call goagain when resumed
        if not self.quit:
            self._meas_running = True
            self.Npts = self.cfg.getintkey('NPTS')
            self.progressBar.setRange(0, self.Npts * 2)
            self.thread = QThread()
            try:
                self.mydvm = Meas(self.mutex, self, self.Npts)
            except Exception as e:
                self._meas_running = False
                self.myprint(f'Instrument error: {e}')
                self.statusBar.showMessage('Instrument error — retrying in 10 s', 10000)
                QTimer.singleShot(10000, self.goagain)
                return
            self.mydvm.moveToThread(self.thread)

            V1, V2, dV, g1, g2 = 5.9, 5.9j, 0.01, 1, 1
            modulation = not self.cbModOff.isChecked()
            self.mydvm.storeV(V1, V2, dV, self.fsig, g1, g2, modulation=modulation)

            self.mydvm.dataReady.connect(self.onNewData)
            self.mydvm.dataSetReady.connect(self.onNewSet)
            self.mydvm.finished.connect(self.thread.quit)
            self.mydvm.finished.connect(self.mydvm.deleteLater)
            self.thread.finished.connect(self.goagain)
            self.stopDVM.connect(self.mydvm.stop)
            self.thread.started.connect(self.mydvm.start)
            self.thread.finished.connect(self.thread.deleteLater)
            self.thread.start()
        else:
            self.loopfinished = True
            self.close()

    # ------------------------------------------------------------------
    # Frequency controls
    # ------------------------------------------------------------------

    def fp(self):
        self.fsig += self.fstep.value()
        self._update_freq_label()

    def fm(self):
        self.fsig -= self.fstep.value()
        self._update_freq_label()

    def _on_tza_gain(self, gain_id):
        self.cfg.setintkey('TZAGAIN', gain_id)
        if self.tza is not None:
            self.tza.set_gain(gain_id)
            self.myprint(f'TZA gain set to {TZA.GAINDICT[gain_id]}')
        else:
            self.myprint('TZA not connected')

    # ------------------------------------------------------------------
    # Exit handling
    # ------------------------------------------------------------------

    def finishUp(self):
        self.statusBar.showMessage('Aborting measurement, please wait...')
        self.myprint("Abort requested — stopping current measurement")
        self.quit = True
        # Set the flag directly on the worker — the signal/slot route doesn't
        # work here because the worker thread's event loop never runs while
        # start() is blocking in its for loop.
        if hasattr(self, 'mydvm'):
            self.mydvm._stop_requested = True
            if self.mydvm._daq_armed:
                try:
                    self.mydvm.dvm.write('ABOR3,(@101:104)')
                    self.mydvm.dvm.write('*CLS')
                except Exception:
                    pass

    def closeEvent(self, event):
        if not self.loopfinished:
            self.statusBar.showMessage('Wait for graceful exit')
            self.myprint("Close event called, but loop not finished")
            self.quit = True
            event.ignore()
        else:
            self.statusBar.showMessage('Quitting')
            self.myprint("Quitting gracefully")
            event.accept()

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    @staticmethod
    def _clear_grid(plots):
        for i in range(2):
            for j in range(2):
                plots[i, j].canvas.ax1.cla()

    @staticmethod
    def _draw_grid(plots):
        for i in range(2):
            for j in range(2):
                plots[i, j].canvas.draw()

    @staticmethod
    def _shade_color(color, t):
        """Interpolates from a light tint of `color` (t=0) to the full,
        saturated color (t=1)."""
        base = np.array(to_rgb(color))
        alpha = 0.15 + 0.85 * t
        return tuple((1 - alpha) * np.ones(3) + alpha * base)

    @staticmethod
    def _plot_switch_points(ax, ptsA, ptsB, color, filled=False, markersize=9):
        """Circles for switch position 0 (straight), squares for switch
        position 1 (cross) -- used for every switch-differentiated scatter
        series. Open (filled=False) for raw data, solid (filled=True) for a
        fitted/predicted overlay, so fit quality is visible at a glance.
        Each series is shaded light-to-dark by sweep order (first point
        lightest, last point full color), so the order is visible and a
        fitted point can be matched to its raw point by matching shade."""
        for pts, marker in ((ptsA, 'o'), (ptsB, 's')):
            n = len(pts)
            for i, p in enumerate(pts):
                t = i / (n - 1) if n > 1 else 1.0
                shade = MainWindow._shade_color(color, t)
                face = shade if filled else 'none'
                ax.plot(np.real(p), np.imag(p), marker=marker, linestyle='None',
                        markerfacecolor=face, markeredgecolor=shade, markersize=markersize)

    def plotraw(self):
        if self.rSet.ts <= 0:
            return
        self._clear_grid(self.rawplots)

        self.t = np.arange(len(self.rSet.Data[0].data))
        Vc  = self.rSet.Data[0].Vc
        phi = -np.angle(-1j * Vc)
        while phi < 0:
            phi += 2 * np.pi
        period = self.rSet.fsamp / self.rSet.fsig
        be1 = phi / (2 * np.pi) * period
        # be1 only phase-aligns within the first period; shift by whole
        # periods so the displayed window sits near the middle of the full
        # acquisition instead of always right at the start.
        n_shift = round((len(self.rSet.Data[0].data) / 2 - be1) / period)
        be = int(be1 + n_shift * period)
        en = int(period * 2) + be

        src0 = self.rSet.Data[0].data
        src1 = self.rSet.Data[1].data
        src2 = self.rSet.Data[2].data

        psa1, f1 = spectral3.mypsa(src0, 1 / self.rSet.fsamp)
        psa2, f2 = spectral3.mypsa(src1, 1 / self.rSet.fsamp)
        psa3, f3 = spectral3.mypsa(src2, 1 / self.rSet.fsamp)

        self.rawplots[0, 0].canvas.ax1.plot(self.t[be:en], self.rSet.Data[0].data[be:en], 'r.')
        self.rawplots[0, 1].canvas.ax1.plot(self.t[be:en], self.rSet.Data[1].data[be:en], 'g.')
        self.rawplots[1, 0].canvas.ax1.plot(self.t[be:en], self.rSet.Data[2].data[be:en], 'b.')
        self.rawplots[1, 1].canvas.ax1.plot(f1, psa1 , 'r-')
        self.rawplots[1, 1].canvas.ax1.plot(f2, psa2 ,   'g-')
        self.rawplots[1, 1].canvas.ax1.plot(f3, psa3,         'b-')
        self.rawplots[0, 0].canvas.ax1.plot(self.t[be:en], self.rSet.Data[0].fv[be:en], 'k-')
        self.rawplots[0, 1].canvas.ax1.plot(self.t[be:en], self.rSet.Data[1].fv[be:en], 'k-')
        self.rawplots[1, 0].canvas.ax1.plot(self.t[be:en], self.rSet.Data[2].fv[be:en], 'k-')
        self.rawplots[1, 1].canvas.ax1.set_xscale('log')
        self.rawplots[1, 1].canvas.ax1.set_yscale('log')
        self.rawplots[1, 1].canvas.ax1.set_title('PSA')

        self._draw_grid(self.rawplots)

    def plotresiduals(self):
        if self.rSet.ts <= 0:
            return
        self._clear_grid(self.residualplots)

        t = np.arange(len(self.rSet.Data[0].data))
        res0 = self.rSet.Data[0].data - self.rSet.Data[0].fv
        res1 = self.rSet.Data[1].data - self.rSet.Data[1].fv
        res2 = self.rSet.Data[2].data - self.rSet.Data[2].fv

        self.residualplots[0, 0].canvas.ax1.plot(t, res0, 'r-', linewidth=0.5)
        self.residualplots[0, 1].canvas.ax1.plot(t, res1, 'g-', linewidth=0.5)
        self.residualplots[1, 0].canvas.ax1.plot(t, res2, 'b-', linewidth=0.5)
        for i, j in ((0, 0), (0, 1), (1, 0)):
            self.residualplots[i, j].canvas.ax1.set_xlabel('sample')
            self.residualplots[i, j].canvas.ax1.set_ylabel('residual')

        psa1, f1 = spectral3.mypsa(res0, 1 / self.rSet.fsamp)
        psa2, f2 = spectral3.mypsa(res1, 1 / self.rSet.fsamp)
        psa3, f3 = spectral3.mypsa(res2, 1 / self.rSet.fsamp)
        self.residualplots[1, 1].canvas.ax1.plot(f1, psa1, 'r-')
        self.residualplots[1, 1].canvas.ax1.plot(f2, psa2, 'g-')
        self.residualplots[1, 1].canvas.ax1.plot(f3, psa3, 'b-')
        self.residualplots[1, 1].canvas.ax1.set_xscale('log')
        self.residualplots[1, 1].canvas.ax1.set_yscale('log')
        self.residualplots[1, 1].canvas.ax1.set_title('residual PSA')

        self._draw_grid(self.residualplots)

    def plotscatter(self):
        self._clear_grid(self.scatterplots)

        self.scatterplots[0, 0].canvas.ax1.set_xlabel('Re(V2)')
        self.scatterplots[0, 0].canvas.ax1.set_ylabel('Im(V2)')
        self.scatterplots[0, 1].canvas.ax1.set_xlabel('Re(V3)')
        self.scatterplots[0, 1].canvas.ax1.set_ylabel('Im(V3)')
        self.scatterplots[1, 0].canvas.ax1.set_xlabel('Re(eta1)')
        self.scatterplots[1, 0].canvas.ax1.set_ylabel('Im(eta1)')
        self.scatterplots[1, 1].canvas.ax1.set_xlabel('Re(eta3)')
        self.scatterplots[1, 1].canvas.ax1.set_ylabel('Im(eta3)')

        # Raw points always come from the live per-point accumulator. It holds
        # exactly the most recently collected cycle's points whether that
        # cycle is still filling up or already finished, so the plot looks
        # the same regardless of how far along the measurement is.
        v2A,   v2B   = np.array(self._partial_v2A),   np.array(self._partial_v2B)
        v3A,   v3B   = np.array(self._partial_v3A),   np.array(self._partial_v3B)
        eta1A, eta1B = np.array(self._partial_eta1A), np.array(self._partial_eta1B)
        eta3A, eta3B = np.array(self._partial_eta3A), np.array(self._partial_eta3B)
        self._plot_switch_points(self.scatterplots[0, 0].canvas.ax1, v2A, v2B, 'm')
        self._plot_switch_points(self.scatterplots[0, 1].canvas.ax1, v3A, v3B, 'c')
        self._plot_switch_points(self.scatterplots[1, 0].canvas.ax1, eta1A, eta1B, 'r')
        self._plot_switch_points(self.scatterplots[1, 1].canvas.ax1, eta3A, eta3B, 'b')

        # Fitted eta1 (solid symbols), so the regression's goodness of fit is
        # visible against the open (raw) eta1 symbols above. Two complex
        # points are the minimum needed to determine the fit, so this updates
        # live from partial data per switch state as soon as each has that
        # many -- it doesn't wait for the cycle (or even that switch state's
        # own sweep) to finish.
        mgain1A, mratio1A = R2FLightAux.fit_eta_regression(eta1A, eta3A)
        eta1A_fit = mgain1A*eta3A - mratio1A if mgain1A is not None else np.array([])
        mgain1B, mratio1B = R2FLightAux.fit_eta_regression(eta1B, eta3B)
        eta1B_fit = mgain1B*eta3B - mratio1B if mgain1B is not None else np.array([])
        self._plot_switch_points(self.scatterplots[1, 0].canvas.ax1, eta1A_fit, eta1B_fit, 'r', filled=True)

        # V2 ellipses (both switch states, top-left only, no ellipses
        # anywhere else) need a full switch-state sweep, so those still come
        # from the completed calc() result -- only shown while self.rData's
        # fit actually matches the points currently on screen, cleared as
        # soon as a new cycle starts collecting.
        if self.rData.Res['ts'] > 0 and self._fit_is_current:
            if self.rData.V2ElliA is not None:
                self.rData.V2ElliA.plot_elli(self.scatterplots[0, 0].canvas.ax1, ellipse_color='m')
            if self.rData.V2ElliB is not None:
                self.rData.V2ElliB.plot_elli(self.scatterplots[0, 0].canvas.ax1, ellipse_color='purple')

        if self.rData.Res['ts'] > 0:
            np.savetxt(os.path.join(self.yyyymmdir, 'eta1.dat'),
                       np.vstack((np.real(self.rData.eta1), np.imag(self.rData.eta1))).T)
            np.savetxt(os.path.join(self.yyyymmdir, 'eta3.dat'),
                       np.vstack((np.real(self.rData.eta3), np.imag(self.rData.eta3))).T)
            np.savetxt(os.path.join(self.yyyymmdir, 'V2.dat'),
                       np.vstack((np.real(self.rData.ave4[:, 1]), np.imag(self.rData.ave4[:, 1]))).T)

        self._draw_grid(self.scatterplots)

    def plotresults(self):
        if self.rData.Res['ts'] <= 0:
            return
        data = self.allData.get_ordered_data()

        t      = data[:, 0] - self.t0
        f      = data[:, 1]
        resist = data[:, 4]
        scale  = np.mean(resist)

        self._clear_grid(self.resultplots)

        self.resultplots[0, 0].canvas.ax1.plot(t, f,      'ko')
        self.resultplots[0, 0].canvas.ax1.set_ylabel('f / Hz')
        self.resultplots[1, 0].canvas.ax1.plot(t, resist, 'b.', alpha=0.5)
        self.resultplots[1, 0].canvas.ax1.set_ylabel('R / Ω')
        if len(resist) >= 7:
            # Savitzky-Golay: window = odd number, ~20% of data, min 7
            wlen = max(7, len(resist) // 5)
            if wlen % 2 == 0:
                wlen += 1
            wlen = min(wlen, len(resist) if len(resist) % 2 == 1 else len(resist) - 1)
            from scipy.signal import savgol_filter
            smooth = savgol_filter(resist, window_length=wlen, polyorder=2)
            self.resultplots[1, 0].canvas.ax1.plot(t, smooth, 'r-', linewidth=1.5)

        if len(resist) > 3:
            s, ad, err = mystat.AllanDeviation(resist)
            self.resultplots[0, 1].canvas.ax1.errorbar(s, ad / scale * 1e6, err / scale * 1e6, fmt='ro')
            self.resultplots[0, 1].canvas.ax1.set_xscale('log')
            self.resultplots[0, 1].canvas.ax1.set_yscale('log')
            self.resultplots[0, 1].canvas.ax1.set_xlabel('time / s')
            self.resultplots[0, 1].canvas.ax1.set_ylabel('rel. Allan dev. / ppm')

        self._draw_grid(self.resultplots)

    def showOutput(self):
        self.output.setText('\n'.join(self.mytext))
        scrollbar = self.output.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def replot(self):
        """Redraws whichever data tab is currently active. Called on every
        incoming data point — must NOT touch the config tab, or it would
        overwrite in-progress edits there."""
        tat = self.tabWidget.master.tabText(self.tabWidget.master.currentIndex())
        if tat == 'raw':
            self.plotraw()
        elif tat == 'scatter':
            self.plotscatter()
        elif tat == 'residuals':
            self.plotresiduals()
        elif tat == 'results':
            self.plotresults()
        elif tat == 'msg':
            self.showOutput()

    def _on_tab_changed(self, index):
        """Called only when the user switches tabs (not on every data point)."""
        tat = self.tabWidget.master.tabText(index)
        if tat == 'config':
            self._refresh_config_fields()
        else:
            self.replot()

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def myprint(self, newtext):
        n  = datetime.datetime.now()
        t  = time.time() - self.t0
        t1 = n.strftime('%H:%M:%S') + f' >{t:8.1f} < : ' + newtext
        t2 = n.strftime('%m/%d') + ' ' + t1 + '\n'
        fn = 'CAP' + n.strftime('%m%d%Y') + '.LOG'
        if self.yyyymmdir:
            with open(os.path.join(self.yyyymmdir, fn), "a") as file:
                file.write(t2)
        self.mytext.append(t1)
        while len(self.mytext) > self.mytextmaxlen:
            self.mytext.pop(0)


class MyTabWidget(QWidget):
    def __init__(self, parent):
        super(QWidget, self).__init__(parent)
        self.master = QTabWidget()
        self.master.resize(300, 200)

        tablabels = ['raw', 'scatter', 'residuals', 'results', 'msg', 'config']
        self.mytabs = [QWidget() for _ in tablabels]
        for tab, label in zip(self.mytabs, tablabels):
            self.master.addTab(tab, label)
        tabs_by_label = dict(zip(tablabels, self.mytabs))

        # tabs that are just a 2x2 grid of plots
        plot_groups = {
            'raw': parent.rawplots,
            'scatter': parent.scatterplots,
            'residuals': parent.residualplots,
            'results': parent.resultplots,
        }
        for label, plots in plot_groups.items():
            glayout = QGridLayout()
            tabs_by_label[label].setLayout(glayout)
            for i in range(2):
                for j in range(2):
                    glayout.addWidget(plots[i, j], i, j)

        msg_layout = QGridLayout()
        tabs_by_label['msg'].setLayout(msg_layout)
        msg_layout.addWidget(parent.output, 0, 0)

        config_layout = QGridLayout()
        tabs_by_label['config'].setLayout(config_layout)
        config_layout.addWidget(parent.cfgWidget, 0, 0)

        layout = QVBoxLayout(self)
        layout.addWidget(self.master)
        self.setLayout(layout)

        self.master.currentChanged.connect(parent._on_tab_changed)


def excepthook(exc_type, exc_value, exc_tb):
    tb  = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    now = datetime.datetime.now()
    print("error caught at", now)
    print("error message:\n", tb)
    with open('R2FNw_errlog.dat', 'w') as fi:
        fi.write("error caught!:\n")
        fi.write("error message:\n" + tb)
    app.quit()


if __name__ == '__main__':
    mutex = QMutex()
    myappid = 'r2fnw.measurement.app.1'
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
    app = QApplication(sys.argv)
    sys.excepthook = excepthook
    window = MainWindow(mutex)
    window.showMaximized()
    app.exec()
    sys.exit()
