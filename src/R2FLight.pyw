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
    QComboBox,
)


class MainWindow(QMainWindow):
    stopDVM = pyqtSignal()

    # auto TZA-gain stepping: bump one gain step whenever V3's amplitude is
    # below this threshold (volts) on every point of a completed cycle, up
    # to gain index _TZA_AUTOSTEP_MAXGAIN (matches the highest gain offered
    # on the UI, V5 = 10 Mohm -- V6/x100,000 is deliberately never auto-selected)
    _TZA_AUTOSTEP_THRESHOLD = 0.3
    _TZA_AUTOSTEP_MAXGAIN = 5

    # gain to start every program run at, regardless of what was left over
    # in the ini from the previous session -- the bridge needs to be roughly
    # balanced before it's safe to run at higher gain, so always start low
    # (V3 = 100 kOhm) and let auto-step/the operator raise it from there
    _TZA_STARTUP_GAIN = 3

    def __init__(self, mutex):
        super().__init__()
        self.quit = False
        self.loopfinished = False
        self._meas_running = False
        # True only once the user has clicked Start -- goagain() no-ops
        # until then, so nothing measures at program launch
        self._armed = False
        # True only while self.rData's fit actually corresponds to the cycle
        # currently being displayed -- cleared the moment a new cycle starts
        # collecting points, so stale fit overlays don't linger on screen.
        self._fit_is_current = False
        # live V1/V3/eta1/eta3 preview accumulated during current ellipse,
        # split by switch state, so the scatter tab looks the same whether a
        # measurement is in progress or complete
        self._partial_v1A = []
        self._partial_v1B = []
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
        self.v2ang = self.cfg.getfloatkey('V2ANG')  # V2 phase relative to V1, degrees (90 = pure quadrature)
        self.v1amp = self.cfg.getfloatkey('V1AMP')  # V1 mean amplitude, volts
        self.v1ang = self.cfg.getfloatkey('V1ANG')  # V1 mean phase, degrees
        self.v2amp = 6.0  # V2 amplitude, volts -- recomputed at Start by _compute_balance_v1v2()
        self.mytext = []
        self.mytextmaxlen = 1000

        dummy_config = CustomData.NPointsConfig(1000, 800000, Nhars=self.cfg.getintkey('NHARS'))
        self.rData   = CustomData.NPoints(dummy_config)
        self.rSet    = CustomData.FourChannels(1000, 800000, 2, [], [], [], [], 0, 0, 0, ts=-1)
        self.allData = CustomData.DAQBuffer(1000, cols=12)

        self.t0 = time.time()
        tnow = time.localtime(self.t0)
        # self.monthdir: where the .LOG file lives (one file per day, but
        # not nested into a day subfolder -- easier to find/tail across a
        # whole month). self.yyyymmdir: where run*.dat/raw*.dat/etc. go,
        # nested per-day since there can be many of them.
        self.monthdir = os.path.join(r'c:\DATA\R2F', time.strftime('%Y%m', tnow))
        if not os.path.isdir(self.monthdir):
            os.mkdir(self.monthdir)
        bd = os.path.join(self.monthdir, time.strftime('%Y%m%d', tnow))
        if not os.path.isdir(bd):
            os.mkdir(bd)
        self.yyyymmdir = bd
        # self.fn/self.rawfn are created by _create_run_files(), called when
        # the user clicks Start -- not here, so the filename timestamp
        # reflects when the measurement actually began, not program launch

        try:
            self.tza = TZA()
        except RuntimeError as e:
            self.tza = None
            print(f'TZA: {e}')

        self._setup_ui()
        self.myprint("Welcome to R2FNw")
        if self.tza is None:
            self.myprint("WARNING: TZA amplifier not found on any COM port")
        self.myprint(f'auto-increase TZA gain: {"ON" if self.cbAutoTZA.isChecked() else "OFF"}')
        self.myprint("Select Reference/DUT above, then click Start")

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

        self.cbModOff = QCheckBox("modulation off")
        self.cbModOff.setChecked(bool(self.cfg.getintkey('MODOFF')))
        self.cbSaveRaw = QCheckBox("save raw points")
        self.cbSaveRaw.setChecked(bool(self.cfg.getintkey('SAVERAW')))
        self.cbAutoV1 = QCheckBox("adjust V1 (null eta3)")
        self.cbAutoV1.setChecked(bool(self.cfg.getintkey('AUTOV1')))
        self.cbAutoTZA = QCheckBox(f"auto-increase TZA gain (V3 < {self._TZA_AUTOSTEP_THRESHOLD:g} V)")
        self.cbAutoTZA.setChecked(bool(self.cfg.getintkey('AUTOTZA')))

        self.buStart = QPushButton("Start")
        self.buStop  = QPushButton("Stop")
        self.buStop.setEnabled(False)

        self.buPause = QPushButton("Pause")
        self.buPause.setCheckable(True)
        self.buPause.setEnabled(False)

        self.buReset = QPushButton("Reset Plots")

        vlayout.addWidget(self.buQuit)
        vlayout.addWidget(self.buStart)
        vlayout.addWidget(self.buStop)
        vlayout.addWidget(self.buPause)
        vlayout.addWidget(self.buReset)
        vlayout.addWidget(self.bufp)
        vlayout.addWidget(self.bufm)
        vlayout.addWidget(self.fstep)
        vlayout.addWidget(self.cbModOff)
        vlayout.addWidget(self.cbSaveRaw)
        vlayout.addWidget(self.cbAutoV1)
        vlayout.addWidget(self.cbAutoTZA)
        vlayout.addWidget(QLabel('current f (frozen):'))
        self.laf = QLabel()
        self._update_freq_label()
        vlayout.addWidget(self.laf)
        vlayout.addWidget(QLabel('current V2 (frozen):'))
        self.laa = QLabel()
        self._update_angle_label()
        vlayout.addWidget(self.laa)
        vlayout.addWidget(QLabel('current V1 (mean):'))
        self.lav = QLabel()
        self._update_v1amp_label()
        vlayout.addWidget(self.lav)
        vlayout.addItem(QSpacerItem(60, 10, QSizePolicy.Fixed, QSizePolicy.Fixed))

        # --- TZA gain radio buttons ---
        vlayout.addWidget(QLabel('TZA gain:'))
        self._tza_gain_group = QButtonGroup(self)
        _tza_labels = {1: '1 k\u03a9', 2: '10 k\u03a9', 3: '100 k\u03a9',
                       4: '1 M\u03a9', 5: '10 M\u03a9'}
        for g in range(1, 6):
            rb = QRadioButton(f'V{g}  ({_tza_labels[g]})')
            self._tza_gain_group.addButton(rb, g)
            vlayout.addWidget(rb)
        self._tza_gain_group.idClicked.connect(self._on_tza_gain)
        # always start low, ignoring whatever gain was persisted from the
        # previous session -- see _TZA_STARTUP_GAIN
        self._apply_tza_gain(self._TZA_STARTUP_GAIN)

        vlayout.addItem(QSpacerItem(60, 10, QSizePolicy.Fixed, QSizePolicy.Fixed))

        # --- REF/DUT component selection (metadata + derived ratio only --
        # neither combo box drives any hardware; the physical component is
        # swapped by hand, this just records what's plugged in) ---
        vlayout.addWidget(QLabel('Reference:'))
        self.cbRef = QComboBox()
        self.cbRef.addItems(list(CustomData.IMPEDANCE_PRESETS.keys()))
        vlayout.addWidget(self.cbRef)

        vlayout.addWidget(QLabel('DUT:'))
        self.cbDut = QComboBox()
        self.cbDut.addItems(list(CustomData.IMPEDANCE_PRESETS.keys()))
        vlayout.addWidget(self.cbDut)

        vlayout.addWidget(QLabel('nominal ratio:'))
        self.laRatio = QLabel()
        vlayout.addWidget(self.laRatio)

        self.cbRef.currentTextChanged.connect(self._apply_ref_preset)
        self.cbDut.currentTextChanged.connect(self._apply_dut_preset)
        # set from the persisted selection with signals blocked (setCurrentText
        # would otherwise fire currentTextChanged only when the persisted value
        # differs from the combo's default first entry), then apply explicitly
        # exactly once below so init always logs/updates consistently
        self.cbRef.blockSignals(True)
        self.cbDut.blockSignals(True)
        self.cbRef.setCurrentText(self.cfg.getstrkey('REFPRESET'))
        self.cbDut.setCurrentText(self.cfg.getstrkey('DUTPRESET'))
        self.cbRef.blockSignals(False)
        self.cbDut.blockSignals(False)
        self._apply_ref_preset(self.cbRef.currentText())
        self._apply_dut_preset(self.cbDut.currentText())

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

        self.laStatusV = QLabel()

        sb_widget = QWidget()
        sb_layout = QHBoxLayout(sb_widget)
        sb_layout.setContentsMargins(0, 0, 0, 0)
        sb_layout.addWidget(self.progressBar)
        sb_layout.addWidget(self.laStatusV)

        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.addPermanentWidget(sb_widget)
        self.statusBar.showMessage('Welcome to R2FNw', 5000)

        # --- signals ---
        self.buQuit.clicked.connect(self.finishUp)
        self.buStart.clicked.connect(self._on_start_clicked)
        self.buStop.clicked.connect(self._on_stop_clicked)
        self.buPause.toggled.connect(self._on_pause_toggled)
        self.buReset.clicked.connect(self._on_reset_clicked)
        self.bufp.clicked.connect(self.fp)
        self.bufm.clicked.connect(self.fm)
        self.cbModOff.toggled.connect(self._on_modoff_toggled)
        self.cbSaveRaw.toggled.connect(self._on_saveraw_toggled)
        self.cbAutoV1.toggled.connect(self._on_autov1_toggled)
        self.cbAutoTZA.toggled.connect(self._on_autotza_toggled)

    def _update_freq_label(self):
        self.laf.setText(f'{self.fsig:8.5f} Hz')
        self.cfg.setfloatkey('FSIG', self.fsig)

    def _update_angle_label(self):
        self.laa.setText(f'{self.v2amp:8.4f} V, {self.v2ang:8.4f} deg')
        self.cfg.setfloatkey('V2ANG', self.v2ang)

    def _v2_complex(self):
        return self.v2amp*np.exp(1j*np.deg2rad(self.v2ang))

    def _update_v1amp_label(self):
        V1c = self.v1amp*np.exp(1j*np.deg2rad(self.v1ang))
        self.lav.setText(f'{V1c.real:+8.4f}{V1c.imag:+8.4f}j   ({self.v1amp:8.4f} V, {self.v1ang:8.4f} deg)')
        self.cfg.setfloatkey('V1AMP', self.v1amp)
        self.cfg.setfloatkey('V1ANG', self.v1ang)

    # keys whose value must come from CustomData.IMPEDANCE_PRESETS -- edited
    # here as free text too (this tab is a raw ini editor), so guard against
    # a typo silently breaking the REFPRESET/DUTPRESET lookups elsewhere
    _CFG_PRESET_KEYS = ('REFPRESET', 'DUTPRESET')

    @staticmethod
    def _cfg_key_kind(key):
        val = R2FConfig.CFG.std[key]
        if isinstance(val, float):
            return 'float'
        if isinstance(val, str):
            return 'str'
        return 'int'

    def _cfg_get(self, key):
        kind = self._cfg_key_kind(key)
        if kind == 'float':
            return self.cfg.getfloatkey(key)
        if kind == 'str':
            return self.cfg.getstrkey(key)
        return self.cfg.getintkey(key)

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
            kind = self._cfg_key_kind(key)
            text = edit.text()

            if kind == 'str':
                if key in self._CFG_PRESET_KEYS and text not in CustomData.IMPEDANCE_PRESETS:
                    self.myprint(f'Config: "{key}" must be one of '
                                 f'{list(CustomData.IMPEDANCE_PRESETS)}, not saved')
                    continue
                self.cfg.setstrkey(key, text)
                edit.setText(text)
                continue

            try:
                val = float(text) if kind == 'float' else int(text)
            except ValueError:
                what = 'a number' if kind == 'float' else 'an integer'
                self.myprint(f'Config: "{key}" must be {what}, not saved')
                continue
            if kind == 'float':
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
        prefactor, alpha_ppm, beta_ppm = self._ratio_prefactor_and_deviation(self.rData.Res["Z"])
        line = [self.rData.Res["ts"], self.rData.Res["fsig"],
                np.real(self.rData.Res["mratio1"]), np.imag(self.rData.Res["mratio1"]),
                self.rData.Res["R"], self.rData.Res["tau"],
                np.real(self.rData.Res["Z"]), np.imag(self.rData.Res["Z"]),
                np.real(self.rData.Res["Y"]), np.imag(self.rData.Res["Y"]),
                alpha_ppm, beta_ppm]

        self._update_freq_label()   # fsig is frozen (from ini) -- just a readout, never reassigned
        self._update_angle_label()  # v2ang is frozen (from ini) -- just a readout, never reassigned

        if self.cbAutoV1.isChecked():
            # eta3_mean=0 is reached where eta1_mean = -mratio1 (see
            # derivation), but eta1 isn't exactly V1/V2 -- there's a real,
            # non-unity gain somewhere in the V1 path (this is why eta1's
            # magnitude was never quite the "expected" value, way back at
            # the start of this project). A one-shot "V1 = -V2*mratio1"
            # jump assumes that gain is 1: if it isn't, it lands on a
            # systematically offset point and can never correct further,
            # since mratio1 doesn't depend on V1 at all -- the one-shot
            # target never changes no matter how wrong the last jump was.
            # Self-calibrate instead: use how far the *previous* V1 (still
            # in self.v1amp/v1ang at this point) actually landed to scale
            # the *next* one, which converges to the true null regardless
            # of what that gain factor is.
            eta1_target   = -self.rData.Res["mratio1"]
            eta1_observed = 0.5*(np.mean(self.rData.eta1A) + np.mean(self.rData.eta1B))
            V1_old = self.v1amp*np.exp(1j*np.deg2rad(self.v1ang))
            V1_target = V1_old * (eta1_target/eta1_observed)
            self.v1amp = np.abs(V1_target)
            self.v1ang = np.degrees(np.angle(V1_target))
        self._update_v1amp_label()

        # gain in effect for the cycle just completed -- captured before any
        # auto-step below, so the raw-file log line reflects the gain the
        # data was actually taken at, not a gain bumped after the fact
        tza_gain = self.cfg.getintkey('TZAGAIN')

        if self.cbAutoTZA.isChecked() and self.tza is not None:
            v3 = self.rData.raw8[:, 2]
            # centered on the ellipse's own mean -- the part that actually
            # carries the eta3 modulation signal, not inflated by whatever
            # DC offset "adjust V1 (null eta3)" hasn't fully nulled out yet
            v3_fluct = np.abs(v3 - np.mean(v3))
            # raw peak (offset + fluctuation) -- what actually risks the DVM
            # rail once gain goes up, regardless of how well-centered it is
            v3_peak = np.max(np.abs(v3))
            if tza_gain < self._TZA_AUTOSTEP_MAXGAIN and np.all(v3_fluct < self._TZA_AUTOSTEP_THRESHOLD):
                if v3_peak * 10 < self._TZA_AUTOSTEP_THRESHOLD:
                    self.myprint(f'V3 (centered) amplitude < {self._TZA_AUTOSTEP_THRESHOLD:g} V on all '
                                  f'points -- auto-increasing TZA gain')
                    self._apply_tza_gain(tza_gain + 1)
                else:
                    self.myprint(f'V3 fluctuation is small but its raw peak ({v3_peak:.4g} V, likely a '
                                  f'not-yet-centered offset) would exceed {self._TZA_AUTOSTEP_THRESHOLD:g} V '
                                  f'after the next gain step -- holding TZA gain until it centers further')

        self.allData.add_point(line)

        try:
            with self._open_with_retry(os.path.join(self.yyyymmdir, self.fn), "a") as file:
                oline = ''.join(f'{item:18.8f} ' for item in line) + '\n'
                file.write(oline)
        except PermissionError as e:
            # a locked file (antivirus/backup scan, etc.) shouldn't be able
            # to take down an unattended run -- skip this line and keep
            # measuring rather than crashing the whole app
            self.myprint(f'WARNING: could not write to {self.fn} ({e}) -- skipping this line')

        # every individual point of the ellipse (both switch states, not
        # switch-averaged), tagged with which switch state it was taken in
        n = len(self.rData.ats)
        switch = (np.arange(n) >= n // 2).astype(float)  # 0 = straight, 1 = cross
        dat = np.vstack((
            self.rData.ats,
            np.ones(n) * self.rData.Res["fsig"],
            switch,
            np.real(self.rData.raw8[:, 0]), np.imag(self.rData.raw8[:, 0]),
            np.real(self.rData.raw8[:, 1]), np.imag(self.rData.raw8[:, 1]),
            np.real(self.rData.raw8[:, 2]), np.imag(self.rData.raw8[:, 2]),
            np.ones(n) * tza_gain)).T

        try:
            with self._open_with_retry(os.path.join(self.yyyymmdir, self.rawfn), "a") as file:
                for l in dat:
                    s = (f'{l[0]:10.0f} {l[1]:18.8f} {l[2]:3.0f} '
                         + ''.join(f'{c:18.8f} ' for c in l[3:-1])
                         + f'{l[-1]:3.0f}' + '\n')
                    file.write(s)
        except PermissionError as e:
            self.myprint(f'WARNING: could not write to {self.rawfn} ({e}) -- skipping these lines')

        eta1A = np.mean(self.rData.eta1A)
        eta3A = np.mean(self.rData.eta3A)
        eta1B = np.mean(self.rData.eta1B)
        eta3B = np.mean(self.rData.eta3B)
        self.myprint('straight: eta1 = {0.real:+10.6f}{0.imag:+10.6f}j   eta3 = {1.real:+10.6f}{1.imag:+10.6f}j'.format(eta1A, eta3A))
        self.myprint('cross:    eta1 = {0.real:+10.6f}{0.imag:+10.6f}j   eta3 = {1.real:+10.6f}{1.imag:+10.6f}j'.format(eta1B, eta3B))
        self.myprint('R = {0:.12e} kOhm   tau = {1:.12e} ns   Z = {2.real:+.6e}{2.imag:+.6e}j Ohm'.format(
            self.rData.Res['R']/1e3, self.rData.Res['tau']*1e9, self.rData.Res['Z']))
        self.myprint(f'ratio = {self._format_prefactor(prefactor)} * (1 + alpha + j*beta)   '
                     f'alpha = {alpha_ppm:+.4f} ppm   beta = {beta_ppm:+.4f} ppm')
        self.replot()
        self.statusBar.showMessage('Data ready, now plotting', 5000)

    def onNewSet(self, MySet: CustomData.FourChannels):
        self.rSet = MySet
        self.progressBar.setValue(MySet.i + 1)
        if MySet.i == 0:
            self._fit_is_current = False
            self._partial_v1A = []
            self._partial_v1B = []
            self._partial_v3A = []
            self._partial_v3B = []
            self._partial_eta1A = []
            self._partial_eta1B = []
            self._partial_eta3A = []
            self._partial_eta3B = []
        # live, per-point preview of V1/V3 (phase-derotated against V2, the
        # constant channel, matching raw8) and eta1/eta3 (V1/V2, V3/V2) as
        # points stream in, ahead of the final switch-averaged/fit values --
        # split by switch state (first Npts points are switch A, next Npts
        # are switch B)
        Vc0 = MySet.Data[0].Vc
        Vc1 = MySet.Data[1].Vc
        if abs(Vc1) > 0:
            cf = np.exp(-1j * np.angle(Vc1))
            v1 = Vc0 * cf
            v3 = MySet.Data[2].Vc * cf
            eta1 = Vc0 / Vc1
            eta3 = MySet.Data[2].Vc / Vc1
            if MySet.i < self.Npts:
                self._partial_v1A.append(v1)
                self._partial_v3A.append(v3)
                self._partial_eta1A.append(eta1)
                self._partial_eta3A.append(eta3)
            else:
                self._partial_v1B.append(v1)
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

    def _on_modoff_toggled(self, checked):
        self.cfg.setintkey('MODOFF', int(checked))

    def _on_saveraw_toggled(self, checked):
        self.cfg.setintkey('SAVERAW', int(checked))

    def _on_autov1_toggled(self, checked):
        self.cfg.setintkey('AUTOV1', int(checked))

    def _on_autotza_toggled(self, checked):
        self.cfg.setintkey('AUTOTZA', int(checked))
        self.myprint(f'auto-increase TZA gain: {"ON" if checked else "OFF"}')

    def _on_vset(self, V1, V2):
        self.laStatusV.setText(
            f'V1 = {V1.real:+8.4f}{V1.imag:+8.4f}j   V2 = {V2.real:+8.4f}{V2.imag:+8.4f}j')

    def goagain(self):
        self._meas_running = False
        if self.buPause.isChecked():
            return   # hold here; _on_pause_toggled will call goagain when resumed
        if self.quit:
            self.loopfinished = True
            self.close()
            return
        if not self._armed:
            return   # stopped -- wait for Start (_on_start_clicked calls goagain())

        self._meas_running = True
        self.Npts = self.cfg.getintkey('NPTS')
        self.progressBar.setRange(0, self.Npts * 2)
        self.thread = QThread()
        try:
            self.mydvm = Meas(self.mutex, self, self.Npts)
        except Exception as e:
            self._meas_running = False
            self.myprint(f'Instrument error: {e}')
            for l in traceback.format_exc().rstrip().splitlines():
                self.myprint('  ' + l)
            self.statusBar.showMessage('Instrument error — retrying in 10 s', 10000)
            QTimer.singleShot(10000, self.goagain)
            return
        self.mydvm.moveToThread(self.thread)

        V1 = self.v1amp*np.exp(1j*np.deg2rad(self.v1ang))
        V2, dV, g1, g2 = self._v2_complex(), 0.01, 1, 1
        modulation = not self.cbModOff.isChecked()
        self.mydvm.storeV(V1, V2, dV, self.fsig, g1, g2, modulation=modulation)

        self.mydvm.dataReady.connect(self.onNewData)
        self.mydvm.dataSetReady.connect(self.onNewSet)
        self.mydvm.vSet.connect(self._on_vset)
        self.mydvm.tzaSaturated.connect(self._on_tza_saturated)
        self.mydvm.finished.connect(self.thread.quit)
        self.mydvm.finished.connect(self.mydvm.deleteLater)
        self.thread.finished.connect(self.goagain)
        self.stopDVM.connect(self.mydvm.stop)
        self.thread.started.connect(self.mydvm.start)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

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
        self._apply_tza_gain(gain_id)

    def _apply_tza_gain(self, gain_id):
        """Set the TZA to gain_id, updating the device, the radio buttons,
        and the persisted config -- shared by manual radio-button clicks,
        the auto-gain-step-up/down paths, and the startup gain reset, so
        every gain change (whatever triggers it) logs the same way."""
        old_gain = self.cfg.getintkey('TZAGAIN')
        self.cfg.setintkey('TZAGAIN', gain_id)
        btn = self._tza_gain_group.button(gain_id)
        if btn is not None:
            btn.setChecked(True)
        if self.tza is not None:
            self.tza.set_gain(gain_id)
            if gain_id != old_gain:
                self.myprint(f'TZA gain changed: V{old_gain} -> V{gain_id} ({TZA.GAINDICT[gain_id]})')
            else:
                self.myprint(f'TZA gain confirmed at V{gain_id} ({TZA.GAINDICT[gain_id]}), unchanged')
        else:
            self.myprint(f'TZA not connected -- gain change to V{gain_id} not applied to hardware')

    def _apply_ref_preset(self, label):
        """Persist the reference-component selection and refresh the derived
        ratio label. No hardware push -- the component is swapped by hand;
        this only feeds Yref (via REFPRESET, read fresh each cycle in
        Meas3.start()) and the run*.dat header."""
        self.cfg.setstrkey('REFPRESET', label)
        ref_type, ref_value = CustomData.IMPEDANCE_PRESETS[label]
        self.myprint(f'Reference set to {label} ({ref_type}, {ref_value:.6g})')
        self._update_ratio_label()

    def _apply_dut_preset(self, label):
        """Persist the DUT-component selection. Metadata/logging + the
        derived ratio only -- the measurement itself doesn't need the DUT's
        nominal value, only the reference's (see _apply_ref_preset)."""
        self.cfg.setstrkey('DUTPRESET', label)
        dut_type, dut_value = CustomData.IMPEDANCE_PRESETS[label]
        self.myprint(f'DUT set to {label} ({dut_type}, {dut_value:.6g})')
        self._update_ratio_label()

    def _update_ratio_label(self):
        # magnitude only (matches the plain "10:1"-style comparison ratio) --
        # via the impedance-based prefactor, not a raw value ratio, since
        # comparing an R's Ohms directly to a C's Farads is meaningless
        # (that was the bug: e.g. REF=1 MOhm vs DUT=10 pF showed "1e+17:1")
        prefactor = self._nominal_ratio_prefactor()
        self.laRatio.setText(f'{abs(prefactor):g}:1')

    def _compute_balance_v1v2(self):
        """Estimate a starting V1/V2 that nulls V3 (I_DUT = I_REF), from the
        nominal REF/DUT component values and fsig -- balance requires
        V1'/V2' = -Z_DUT/Z_REF (see derivation), and V1'~=V1, V2'~=V2 is a
        good enough starting approximation (the "adjust V1 (null eta3)"
        feedback refines it further from real data once cycles start).
        V2 starts at 6+0j V; if that implies |V1| > 10 V, V2 is stepped down
        by 10x and V1 recomputed, until |V1| < 10 V."""
        ref_type, ref_value = CustomData.IMPEDANCE_PRESETS[self.cfg.getstrkey('REFPRESET')]
        dut_type, dut_value = CustomData.IMPEDANCE_PRESETS[self.cfg.getstrkey('DUTPRESET')]
        z_ref = CustomData.nominal_impedance(ref_type, ref_value, self.fsig)
        z_dut = CustomData.nominal_impedance(dut_type, dut_value, self.fsig)

        v2amp = 6.0
        for _ in range(10):
            V2 = complex(v2amp, 0.0)
            V1 = -V2 * (z_dut / z_ref)
            if abs(V1) < 10.0:
                break
            v2amp /= 10.0
        else:
            self.myprint('WARNING: could not bring |V1| under 10 V after 10 reductions of V2 '
                          '-- using the last estimate anyway')

        self.v2amp = v2amp
        self.v2ang = 0.0
        self.v1amp = abs(V1)
        self.v1ang = np.degrees(np.angle(V1))
        self.myprint(f'Balance estimate: V1 = {V1.real:+.4f}{V1.imag:+.4f}j V '
                     f'({self.v1amp:.4f} V, {self.v1ang:.4f} deg)   '
                     f'V2 = {V2.real:+.4f}{V2.imag:+.4f}j V')
        self._update_v1amp_label()
        self._update_angle_label()

    def _nominal_ratio_prefactor(self):
        """The clean nominal design ratio (10, j, 0.1j, ... via
        CustomData.snap_ratio_prefactor) for the currently selected REF/DUT
        presets at the current fsig."""
        ref_type, ref_value = CustomData.IMPEDANCE_PRESETS[self.cfg.getstrkey('REFPRESET')]
        dut_type, dut_value = CustomData.IMPEDANCE_PRESETS[self.cfg.getstrkey('DUTPRESET')]
        z_ref = CustomData.nominal_impedance(ref_type, ref_value, self.fsig)
        z_dut_nominal = CustomData.nominal_impedance(dut_type, dut_value, self.fsig)
        return CustomData.snap_ratio_prefactor(z_dut_nominal / z_ref)

    def _ratio_prefactor_and_deviation(self, Z_measured):
        """Express Z_measured/Z_REF_nominal as prefactor*(1+alpha+j*beta):
        prefactor is the clean nominal design ratio, alpha/beta (ppm) are
        how far the actual measurement deviates from that nominal ratio."""
        ref_type, ref_value = CustomData.IMPEDANCE_PRESETS[self.cfg.getstrkey('REFPRESET')]
        z_ref = CustomData.nominal_impedance(ref_type, ref_value, self.fsig)
        prefactor = self._nominal_ratio_prefactor()

        measured_ratio = Z_measured / z_ref
        deviation = measured_ratio / prefactor - 1
        alpha_ppm = deviation.real * 1e6
        beta_ppm  = deviation.imag * 1e6
        return prefactor, alpha_ppm, beta_ppm

    @staticmethod
    def _format_prefactor(p):
        return f'{p.real:g}' if p.imag == 0 else f'{p.imag:g}j'

    @staticmethod
    def _open_with_retry(path, mode, max_attempts=5, delay=0.2):
        """Open path in mode (ascii text), retrying briefly on a transient
        PermissionError -- e.g. antivirus/backup software momentarily
        locking the file, which happened during an unattended overnight run
        and crashed the whole app on the first attempt. Raises the last
        PermissionError if it never clears."""
        for attempt in range(1, max_attempts + 1):
            try:
                return open(path, mode, encoding="ascii")
            except PermissionError:
                if attempt == max_attempts:
                    raise
                time.sleep(delay)

    def _create_run_files(self):
        """Create the timestamped run*.dat/raw*.dat files and write their
        headers. Called from _on_start_clicked so the filename timestamp
        reflects when the measurement actually began, not program launch."""
        tnow = time.localtime()
        self.fn    = 'run' + time.strftime('%H%M%S', tnow) + '.dat'
        self.rawfn = 'raw' + time.strftime('%H%M%S', tnow) + '.dat'

        ref_preset = self.cfg.getstrkey('REFPRESET')
        dut_preset = self.cfg.getstrkey('DUTPRESET')
        ref_type, ref_value = CustomData.IMPEDANCE_PRESETS[ref_preset]
        dut_type, dut_value = CustomData.IMPEDANCE_PRESETS[dut_preset]
        prefactor = self._nominal_ratio_prefactor()
        setup_header = (f'# REF: {ref_preset} ({ref_type}, {ref_value:.6e})   '
                         f'DUT: {dut_preset} ({dut_type}, {dut_value:.6e}, nominal)   '
                         f'ratio: {abs(prefactor):g}:1   '
                         f'result = {self._format_prefactor(prefactor)}*(1+alpha+j*beta), '
                         f'alpha/beta in ppm\n')

        with self._open_with_retry(os.path.join(self.yyyymmdir, self.fn), "w") as file:
            file.write(setup_header)
            file.write('#' + ''.join(f'{h:>18} ' for h in
                        ['ts', 'fsig/Hz', 'Re(mratio1)', 'Im(mratio1)', 'R/Ohm', 'tau/s',
                         'Re(Z)', 'Im(Z)', 'Re(Y)', 'Im(Y)', 'alpha/ppm', 'beta/ppm']) + '\n')
        with self._open_with_retry(os.path.join(self.yyyymmdir, self.rawfn), "w") as file:
            file.write(setup_header)
            file.write('#' + f'{"ts":>9} ' + f'{"fsig/Hz":>18} ' + f'{"switch":>3} '
                        + ''.join(f'{h:>18} ' for h in
                        ['Re(V1)', 'Im(V1)', 'Re(V2)', 'Im(V2)', 'Re(V3)', 'Im(V3)'])
                        + f'{"TZAgain":>3}'
                        + '  (switch: 0=straight, 1=cross; TZAgain: 1-6 = V1-V6)\n')

    def _on_start_clicked(self):
        if self._armed:
            return
        self._armed = True

        ref_preset = self.cfg.getstrkey('REFPRESET')
        dut_preset = self.cfg.getstrkey('DUTPRESET')
        ref_type, _ = CustomData.IMPEDANCE_PRESETS[ref_preset]
        dut_type, _ = CustomData.IMPEDANCE_PRESETS[dut_preset]
        banner = '#' * 70
        self.myprint(banner)
        self.myprint(f'REF: {ref_preset} ({ref_type})   DUT: {dut_preset} ({dut_type})')
        self.myprint(banner)

        self._compute_balance_v1v2()
        self._create_run_files()
        self.buStart.setEnabled(False)
        self.buStop.setEnabled(True)
        self.buPause.setEnabled(True)
        self.cbRef.setEnabled(False)
        self.cbDut.setEnabled(False)
        self.myprint(f'Measurement started -- data: {os.path.join(self.yyyymmdir, self.fn)}   '
                     f'raw: {os.path.join(self.yyyymmdir, self.rawfn)}')
        self.statusBar.showMessage('Measurement running')
        self.goagain()

    def _on_stop_clicked(self):
        if not self._armed:
            return
        self._armed = False
        self.myprint('Stop requested -- aborting current cycle')
        self.statusBar.showMessage('Stopped')
        if hasattr(self, 'mydvm'):
            self.mydvm._stop_requested = True
            if self.mydvm._daq_armed:
                try:
                    self.mydvm.dvm.write('ABOR3,(@101:104)')
                    self.mydvm.dvm.write('*CLS')
                except Exception:
                    pass
        # reset pause state without triggering _on_pause_toggled (which would
        # log "Resumed" and re-call goagain() -- harmless since goagain()
        # now no-ops while not armed, but confusing in the log)
        self.buPause.blockSignals(True)
        self.buPause.setChecked(False)
        self.buPause.blockSignals(False)
        self.buPause.setText("Pause")
        self.buPause.setEnabled(False)
        self.buStart.setEnabled(True)
        self.buStop.setEnabled(False)
        self.cbRef.setEnabled(True)
        self.cbDut.setEnabled(True)

    def _on_reset_clicked(self):
        """Clears the accumulated results-tab history (f/alpha/beta/Allan
        deviation) -- e.g. to drop the initial gain-hunting/settling
        transient from the view. Doesn't touch run*.dat/raw*.dat (those keep
        the full record) or interrupt a measurement in progress."""
        self.allData = CustomData.DAQBuffer(1000, cols=12)
        self.myprint('Results plot history reset')
        self.replot()

    def _on_tza_saturated(self, ptp):
        """Slot for Meas.tzaSaturated (queued across threads) -- the cycle
        that triggered this has already been discarded by Meas (dataReady
        wasn't emitted for it), so just drop the gain a notch here; goagain()
        starts the next cycle once Meas.finished fires."""
        current_gain = self.cfg.getintkey('TZAGAIN')
        if current_gain > 1:
            self._apply_tza_gain(current_gain - 1)
        else:
            self.myprint('TZA already at minimum gain (V1) -- cannot reduce further')

    # ------------------------------------------------------------------
    # Exit handling
    # ------------------------------------------------------------------

    def finishUp(self):
        self.quit = True
        if not self._armed:
            # Never started (or already stopped) -- goagain() isn't going to
            # fire again on its own to notice self.quit, so finalize directly
            # (mirrors what goagain() does once an in-flight cycle unwinds).
            self.loopfinished = True
            self.close()
            return
        self.statusBar.showMessage('Aborting measurement, please wait...')
        self.myprint("Abort requested — stopping current measurement")
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

        self.scatterplots[0, 0].canvas.ax1.set_xlabel('Re(V1)')
        self.scatterplots[0, 0].canvas.ax1.set_ylabel('Im(V1)')
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
        v1A,   v1B   = np.array(self._partial_v1A),   np.array(self._partial_v1B)
        v3A,   v3B   = np.array(self._partial_v3A),   np.array(self._partial_v3B)
        eta1A, eta1B = np.array(self._partial_eta1A), np.array(self._partial_eta1B)
        eta3A, eta3B = np.array(self._partial_eta3A), np.array(self._partial_eta3B)
        self._plot_switch_points(self.scatterplots[0, 0].canvas.ax1, v1A, v1B, 'm')
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

        # V1 ellipses (both switch states, top-left only, no ellipses
        # anywhere else) need a full switch-state sweep, so those still come
        # from the completed calc() result -- only shown while self.rData's
        # fit actually matches the points currently on screen, cleared as
        # soon as a new cycle starts collecting.
        if self.rData.Res['ts'] > 0 and self._fit_is_current:
            if self.rData.V1ElliA is not None:
                self.rData.V1ElliA.plot_elli(self.scatterplots[0, 0].canvas.ax1, ellipse_color='m')
            if self.rData.V1ElliB is not None:
                self.rData.V1ElliB.plot_elli(self.scatterplots[0, 0].canvas.ax1, ellipse_color='purple')

        if self.rData.Res['ts'] > 0:
            np.savetxt(os.path.join(self.yyyymmdir, 'eta1.dat'),
                       np.vstack((np.real(self.rData.eta1), np.imag(self.rData.eta1))).T)
            np.savetxt(os.path.join(self.yyyymmdir, 'eta3.dat'),
                       np.vstack((np.real(self.rData.eta3), np.imag(self.rData.eta3))).T)
            np.savetxt(os.path.join(self.yyyymmdir, 'V1.dat'),
                       np.vstack((np.real(self.rData.ave4[:, 0]), np.imag(self.rData.ave4[:, 0]))).T)

        self._draw_grid(self.scatterplots)

    def plotresults(self):
        if self.rData.Res['ts'] <= 0:
            return
        data = self.allData.get_ordered_data()

        t     = data[:, 0] - self.t0
        f     = data[:, 1]
        alpha = data[:, 10]
        beta  = data[:, 11]

        self._clear_grid(self.resultplots)

        self.resultplots[0, 0].canvas.ax1.plot(t, f,      'ko')
        self.resultplots[0, 0].canvas.ax1.set_ylabel('f / Hz')
        self.resultplots[1, 0].canvas.ax1.plot(t, alpha, 'b.', alpha=0.5)
        self.resultplots[1, 0].canvas.ax1.set_ylabel('alpha / ppm')
        self.resultplots[1, 1].canvas.ax1.plot(t, beta,  'g.', alpha=0.5)
        self.resultplots[1, 1].canvas.ax1.set_ylabel('beta / ppm')
        if len(alpha) >= 7:
            # Savitzky-Golay: window = odd number, ~20% of data, min 7
            wlen = max(7, len(alpha) // 5)
            if wlen % 2 == 0:
                wlen += 1
            wlen = min(wlen, len(alpha) if len(alpha) % 2 == 1 else len(alpha) - 1)
            from scipy.signal import savgol_filter
            smooth = savgol_filter(alpha, window_length=wlen, polyorder=2)
            self.resultplots[1, 0].canvas.ax1.plot(t, smooth, 'r-', linewidth=1.5)

        if len(alpha) > 3:
            # alpha/beta are already fractional (ppm) deviations, so their
            # Allan deviations are already in ppm -- no rescaling by a mean
            # needed (unlike the old R-based version, where R was absolute)
            s_a, ad_a, err_a = mystat.AllanDeviation(alpha)
            s_b, ad_b, err_b = mystat.AllanDeviation(beta)
            self.resultplots[0, 1].canvas.ax1.errorbar(s_a, ad_a, err_a, fmt='bo', label='alpha')
            self.resultplots[0, 1].canvas.ax1.errorbar(s_b, ad_b, err_b, fmt='go', label='beta')
            self.resultplots[0, 1].canvas.ax1.set_xscale('log')
            self.resultplots[0, 1].canvas.ax1.set_yscale('log')
            self.resultplots[0, 1].canvas.ax1.set_xlabel('time / s')
            self.resultplots[0, 1].canvas.ax1.set_ylabel('rel. Allan dev. / ppm')
            self.resultplots[0, 1].canvas.ax1.legend()

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
        fn = 'R2FLight_' + n.strftime('%m%d%Y') + '.LOG'
        if self.monthdir:
            with open(os.path.join(self.monthdir, fn), "a") as file:
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
    try:
        banner = '!' * 70
        window.myprint(banner)
        window.myprint('UNHANDLED EXCEPTION -- quitting')
        for l in tb.rstrip().splitlines():
            window.myprint('  ' + l)
        window.myprint(banner)
        target_dir = window.yyyymmdir
    except Exception:
        target_dir = '.'
    fn = 'R2FNw_errlog_' + now.strftime('%Y%m%d_%H%M%S') + '.dat'
    with open(os.path.join(target_dir, fn), 'w') as fi:
        fi.write(f"error caught at {now}:\n")
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
