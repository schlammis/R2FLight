
from PyQt5.QtCore import (
QObject,
pyqtSignal,
pyqtSlot)

import time
import os
import threading
import traceback
import numpy as np
import pyvisa
import R2FLightAux
import CustomData
import re


class TZASaturated(Exception):
    """Raised when a fetched V3 waveform indicates saturation -- either no
    fluctuation (TZA/DVM input pinned at a rail) or a DVM overload code
    (value pinned at an absurd magnitude instead of a real reading)."""
    def __init__(self, value):
        self.value = value
        super().__init__(f'V3 appears saturated (value={value:.6g})')


class Meas(QObject):
    finished = pyqtSignal()
    dataReady  = pyqtSignal(CustomData.NPoints)
    dataSetReady  = pyqtSignal(CustomData.FourChannels)
    vSet  = pyqtSignal(complex, complex)
    tzaSaturated = pyqtSignal(float)  # V3 ptp, in volts -- picked up on the GUI thread

    # a saturated/pinned channel reads an essentially constant value -- but
    # so does a genuinely well-balanced V3 (the bridge is *supposed* to null
    # it close to zero). Peak-to-peak alone can't tell those apart, so
    # "no fluctuation" only counts as saturation when the level is ALSO
    # sitting near the rail rather than near zero -- V3_FULLSCALE matches
    # the ACQ3:VOLT 0.3 range programmed onto V3's channel in precmd().
    V3_SATURATION_PTP = 0.001
    V3_FULLSCALE = 0.3
    V3_SATURATION_LEVEL_FRAC = 0.5

    # the DVM reports an out-of-range/overload reading as a huge sentinel
    # value (observed ~1e17 on this hardware) instead of a real voltage --
    # V3's full-scale is 0.3 V, so anything even remotely this large can only
    # be an overload code, never a genuine reading
    V3_OVERFLOW_ABS = 1e6

    def __init__(self,mutex,parent,NDpts):
        super(QObject, self).__init__()
        self.NDpts = NDpts  # number of points in one full ellipse sweep (one switch position)
        self.Npts = 2*NDpts # NDpts points in switch position A, then NDpts more in switch position B
        self.par =parent
        self.isidle =True
        self.mutex       = mutex
        self.co=-1
        self.runt = time.time()
        self.fsig = 1000
        self.fsamp = int(800000/16)
        self.maxco=1200*2
        self._stop_requested = False
        self.modulation = True
        self._connect_to_instruments()
        self.oldf=-1
        self.precmd()

    def _connect_to_instruments(self):
        """Open the VISA resources with a hard watchdog timeout. Some VISA
        backends let open_resource() hang indefinitely if an instrument is
        in a wedged state (e.g. after a prior I/O fault) -- a resource's own
        .timeout only bounds reads/writes *after* it's open, not the open()
        call itself. Running the connection attempt in a background thread
        and joining with a timeout guarantees this fails loudly and
        promptly instead of freezing the GUI thread forever (Meas() is
        always constructed on the GUI thread, before moveToThread())."""
        result = {}

        def _connect():
            try:
                rm = pyvisa.ResourceManager()
                sg1_pattern = "^USB.*:MY62.*"
                dvm_pattern = "^USB.*:MY59.*"
                sg1_name = ''
                dvm_name = ''
                for i in rm.list_resources():
                    if re.search(sg1_pattern, i):
                        sg1_name = i
                    elif re.search(dvm_pattern, i):
                        dvm_name = i
                # open_resource('') on a not-found instrument can also hang
                # on some backends -- fail fast instead
                if not sg1_name:
                    raise RuntimeError(f'signal generator not found (no VISA resource matching {sg1_pattern!r})')
                if not dvm_name:
                    raise RuntimeError(f'DVM not found (no VISA resource matching {dvm_pattern!r})')
                result['rm']  = rm
                result['sg1'] = rm.open_resource(sg1_name)
                result['dvm'] = rm.open_resource(dvm_name)
            except Exception as e:
                result['error'] = e

        t = threading.Thread(target=_connect, daemon=True)
        t.start()
        t.join(timeout=10.0)
        if t.is_alive():
            raise RuntimeError('Timed out connecting to instruments after 10 s -- '
                                'an instrument may be in a stuck state (try power-cycling it)')
        if 'error' in result:
            raise result['error']
        self.rm, self.sg1, self.dvm = result['rm'], result['sg1'], result['dvm']

    def precmd(self):
        self.sg1.write('UNIT:ANGL DEG')
        self.sg1.write('VOLT:UNIT VPP')
        self.sg1.write('VOLT:OFFS 0')
        self.sg1.write('SOUR1:FUNC SIN')
        self.sg1.write('SOUR2:FUNC SIN')
        self.sg1.write('OUTP1 ON')
        self.sg1.write('OUTP2 ON')
        self.dvm.timeout=25000
        self.dvm.write('FORM3 REAL')
        #self.dvm.write('ACQ3:VOLT 3,DIFF,AC,TIME,(@101:102)')
        #self.dvm.write('ACQ3:VOLT 0.3,DIFF,AC,TIME,(@103:104)')
        self.dvm.write('ACQ3:VOLT 18,SEND,DC,TIME,(@101:102)')
        self.dvm.write('ACQ3:VOLT 0.3,SEND,DC,TIME,(@103:104)')

        self.sampcount = self.par.cfg.getintkey('SAMPCOUNT')
        self.settle_time = self.par.cfg.getfloatkey('SETTLE')
        self.save_raw = bool(self.par.cfg.getintkey('SAVERAW'))
        self.dvm.write('SAMP3:RATE {0:8.2f},(@101:104)'.format(self.fsamp))
        self.dvm.write('SAMP3:COUN {0},(@101:104)'.format(self.sampcount))
        self.dvm.write('INP3:COUP AC,(@101:104)')
        self.dvm.write('TRIG3:SOUR BUS,(@101:104)')

    def write1dbg(self,ostr,debug=False):
        self.sg1.write(ostr)
        if debug:
            print(ostr)
        time.sleep(0.001)
 
    def storeV(self,V1c,V2c,dV1,fsig,g1,g2,modulation=True):
        if self.isidle:
            self.V1c=V1c
            self.V2c=V2c
            self.dV1 = dV1
            self.fsig = fsig
            self.g1 = g1
            self.g2 = g2
            self.modulation = modulation

    def prepForMeas(self):
        if self.co==0:
            self.par.myprint("V1= {0:8.3f}  V2={1:8.3f} dV1={2:8.3f}  f={3:8.5f} kHz".format(self.V1c,self.V2c,self.dV1,self.fsig/1000))
        if self.co<self.NDpts:
            self.dvm.write('ROUT:OPEN (@211,248)')
            self.dvm.write('ROUT:CLOS (@218,241)') # 8->1 1->4
        else:
            self.dvm.write('ROUT:OPEN (@218,241)')
            self.dvm.write('ROUT:CLOS (@211,248)') # 1->1 8->4
        if self.modulation:
            # V2 (the C_REF branch) is held perfectly constant; V1 (the
            # Z_DUT branch) traces the small ellipse that gives eta1/eta3
            # enough independent variation for the regression fit.
            ang = (self.co%self.NDpts)/self.NDpts*2*np.pi
            a = self.dV1*1.2
            b = self.dV1*0.6
            theta = 30/180*np.pi
            self.V1 = self.V1c+np.exp(1j*theta)*(a*np.cos(ang)+1j*b*np.sin(ang))
        else:
            self.V1 = self.V1c
        self.V2 = self.V2c
        V1amp   = np.abs(self.V1)
        V2amp   = np.abs(self.V2)
        V1phase = np.angle(self.V1)/np.pi*180
        V2phase = np.angle(self.V2)/np.pi*180
        self.write1dbg('SOUR1:VOLT {0:8.4f}'.format(V1amp))
        self.write1dbg('SOUR2:VOLT {0:8.4f}'.format(V2amp))
        self.write1dbg('SOUR1:FREQ {0:8.4f}'.format(self.fsig))
        self.write1dbg('SOUR2:FREQ {0:8.4f}'.format(self.fsig))
        self.write1dbg('PHAS:SYNC')
        self.write1dbg('SOUR1:PHASE {0:8.4f}'.format(V1phase))
        self.write1dbg('SOUR2:PHASE {0:8.4f}'.format(V2phase))
        V1=float(self.sg1.query('SOUR1:VOLT?'))
        V2=float(self.sg1.query('SOUR2:VOLT?'))
        phase1=float(self.sg1.query('SOUR1:PHASE?'))
        phase2=float(self.sg1.query('SOUR2:PHASE?'))
        self.V1rb = V1 * np.exp(1j*phase1/180*np.pi)
        self.V2rb = V2 * np.exp(1j*phase2/180*np.pi)
        self.vSet.emit(self.V1rb, self.V2rb)
        ret=self.sg1.query('SYSTem:ERRor?')
        retval = int(ret.split(',')[0])
        if retval!=0:
            self.par.myprint(f'Error: {ret}')


        
    def sendtrig(self):
        self.dvm.write('INIT3 (@101:104)')# %%
        self.dvm.write('*TRG')

    def _wait(self, duration):
        """Sleeps for `duration` seconds, polling _stop_requested so an abort
        isn't delayed by a long wait."""
        t_end = time.time() + duration
        while time.time() < t_end:
            if self._stop_requested:
                break
            time.sleep(0.01)

    @pyqtSlot()
    def start(self):
        self.isidle=False
        start = time.time()
        ref_type, ref_value = CustomData.IMPEDANCE_PRESETS[self.par.cfg.getstrkey('REFPRESET')]
        my_config = CustomData.NPointsConfig(
            fsig=self.fsig,
            fsamp=self.fsamp,
            g1=self.g1,
            g2=self.g2,
            N=self.Npts,
            Nhars=self.par.cfg.getintkey('NHARS'),
            ref_type=ref_type,
            ref_value=ref_value,
            modulation=self.modulation
        )
        self._stop_requested = False
        self._daq_armed = False
        self._point_failed = False
        self.rawN = CustomData.NPoints(my_config)
        for self.co in range(self.Npts):
            if self._stop_requested:
                break
            self._last_ch = None
            try:
                self.prepForMeas()
                # let the relay switch and the signal generator's amplitude/phase
                # step settle before triggering, so that transient doesn't leak
                # into the acquired data
                self._wait(self.settle_time)
                if self._stop_requested:
                    break
                self.sendtrig()
                self._daq_armed = True
                # the DAQ needs sampcount/fsamp seconds to actually finish
                # acquiring; fetching before that returns stale/incomplete
                # data (and the resulting bad fit just gets retried forever
                # by goagain(), since the mismatch never goes away on its
                # own) -- so derive the wait from the real acquisition time
                # instead of assuming it, with a margin for trigger/VISA
                # overhead, never less generous than the old fixed 1.0 s
                acq_time = self.sampcount / self.fsamp
                self._wait(max(1.0, acq_time * 1.2 + 0.3))
                if self._stop_requested:
                    break
                self._daq_armed = False
                self.getvals()
            except TZASaturated as e:
                # V3 is saturated (pinned flat, or an overload code) -- the
                # fit on this cycle would be garbage regardless of how many
                # more points we take, so discard it immediately, drop the
                # TZA gain a notch, and let goagain() start a fresh cycle at
                # the lower gain.
                self._daq_armed = False
                self._handle_tza_saturation(e.value)
                self._point_failed = True
                break
            except Exception:
                # Don't let one bad point (glitchy sample, VISA hiccup, fit
                # that fails to converge) take down an unattended overnight
                # run -- log it verbosely and abort just this cycle; goagain()
                # picks back up with a fresh cycle.
                self._daq_armed = False
                self._log_point_failure()
                self._point_failed = True
                break
        self.isidle = True
        if not self._stop_requested and not self._point_failed:
            try:
                self.rawN.calc()
                self.dataReady.emit(self.rawN)
            except Exception:
                self._log_point_failure(where='calc() after full cycle')
        self.finished.emit()

    def _log_point_failure(self, where=None):
        tb = traceback.format_exc()
        switch = 'straight' if self.co < self.NDpts else 'cross'
        label = where or f'point co={self.co} ({switch})'
        self.par.myprint(f'Measurement failed at {label} -- aborting this cycle, retrying next cycle:')
        for name, ch in zip(('ch1', 'ch2', 'ch3', 'ch4'), self._last_ch or ()):
            arr = np.asarray(ch, dtype=float)
            nbad = int(np.sum(~np.isfinite(arr)))
            finite = arr[np.isfinite(arr)]
            if finite.size:
                self.par.myprint(f'  {name}: n={arr.size} nonfinite={nbad} '
                                  f'min={finite.min():.6g} max={finite.max():.6g} mean={finite.mean():.6g}')
            else:
                self.par.myprint(f'  {name}: n={arr.size} nonfinite={nbad} (all non-finite)')
        for l in tb.rstrip().splitlines():
            self.par.myprint('  ' + l)

    def _handle_tza_saturation(self, value):
        # Runs on the worker thread -- log here (myprint just appends to a
        # list/file, thread-safe), but leave the actual gain change to the
        # GUI thread via the queued tzaSaturated signal, since it touches
        # the TZA gain radio buttons.
        switch = 'straight' if self.co < self.NDpts else 'cross'
        self.par.myprint(f'V3 saturated at co={self.co} ({switch}) '
                          f'-- discarding this cycle and reducing TZA gain')
        self.tzaSaturated.emit(value)

    @pyqtSlot()
    def stop(self):
        self._stop_requested = True
        if self._daq_armed:
            try:
                self.dvm.write('ABOR3,(@101:104)')
                self.dvm.write('*CLS')   # flush output queue so next session starts clean
            except Exception:
                pass

    def getvals(self):
        if self.co<self.NDpts:
            ch1=self.dvm.query_binary_values('FETCH3? (@101)',  datatype='f', is_big_endian=True)
            ch2=self.dvm.query_binary_values('FETCH3? (@102)',  datatype='f', is_big_endian=True)
        else:
            ch2=self.dvm.query_binary_values('FETCH3? (@101)',  datatype='f', is_big_endian=True)
            ch1=self.dvm.query_binary_values('FETCH3? (@102)',  datatype='f', is_big_endian=True)
        ch3=self.dvm.query_binary_values('FETCH3? (@103)',  datatype='f', is_big_endian=True)
        ch4=self.dvm.query_binary_values('FETCH3? (@104)',  datatype='f', is_big_endian=True)
        self._last_ch = (ch1, ch2, ch3, ch4)  # for diagnostics if setPoint's fit fails below

        v3maxabs = float(np.max(np.abs(ch3)))
        if v3maxabs > self.V3_OVERFLOW_ABS:
            switch = 'straight' if self.co < self.NDpts else 'cross'
            self.par.myprint(f'V3 OVERFLOW at co={self.co} ({switch}): max|V3|={v3maxabs:.6g} '
                              f'(DVM overload code, not a real reading)')
            # don't let the overload code reach setPoint()/the fit -- it would
            # poison this cycle's eta3/R and, worse, feed a garbage mratio1
            # into the "adjust V1 (null eta3)" feedback, which then walks V1
            # further off null every cycle instead of recovering
            raise TZASaturated(v3maxabs)

        v3ptp = float(np.ptp(ch3))
        v3level = float(np.mean(np.abs(ch3)))
        if (v3ptp < self.V3_SATURATION_PTP
                and v3level > self.V3_FULLSCALE * self.V3_SATURATION_LEVEL_FRAC):
            switch = 'straight' if self.co < self.NDpts else 'cross'
            self.par.myprint(f'V3 NO FLUCTUATION at co={self.co} ({switch}): ptp={v3ptp:.6g} V, '
                              f'level={v3level:.6g} V (pinned near the rail, not a real reading)')
            raise TZASaturated(v3ptp)

        self.rawN.setPoint(self.co,ch1,ch2,ch3,ch4,self.V1rb,self.V2rb,time.time())
        self.dataSetReady.emit(self.rawN.Data[self.co])
        if self.save_raw and (self.co == self.NDpts-1 or self.co == self.Npts-1):
            self._saveRawPoint(ch1,ch2,ch3)

    def _saveRawPoint(self,ch1,ch2,ch3):
        """Dump the raw (unfitted) samples of one point per completed ellipse,
        for offline noise analysis."""
        fn = 'raw3ch' + time.strftime('%H%M%S') + '.npy'
        np.save(os.path.join(self.par.yyyymmdir, fn), np.vstack((ch1,ch2,ch3)))
