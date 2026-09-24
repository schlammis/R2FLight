"""
Standalone two-channel raw sampler with relay switching.

Samples V1/V2 (DAQ channels 101/102, the ones affected by the straight/cross
relay) repeatedly, flipping the relay back and forth, and writes the raw
(unfitted) waveforms to a single .npz file.

This reuses Meas3.Meas directly for the connection, SCPI setup, and
signal-generator/relay-switch logic (_connect_to_instruments, precmd,
prepForMeas, sendtrig) -- NOT a hand-copied reimplementation -- so every
setting (SAMPCOUNT, SETTLE, ACQ3:VOLT ranges, fsamp, V1/V2 amplitude and
phase) is guaranteed identical to whatever R2FLight.pyw/Meas3.py currently
use, read from the same R2FLight.ini. It skips the full ellipse-modulation
+ regression-fit pipeline in Meas.start(), since this tool just wants raw
channel data with the relay flipped, not a Z_DUT measurement.

Usage:
    python standalone_sampler.py [--npoints N] [--points-per-switch M] [--outdir DIR]

    --npoints            total points to sample (default 20)
    --points-per-switch  how many consecutive points before flipping the
                          relay (default 1 -- alternates every point)
    --outdir             where to write the .npz file (default
                          C:\\DATA\\R2F\\standalone)
"""
import argparse
import os
import sys
import time

import numpy as np
from PyQt5.QtCore import QCoreApplication, QMutex

import R2FConfig
from Meas3 import Meas


class _MinimalParent:
    """Just enough of MainWindow's interface for Meas3.Meas to run outside
    the GUI: .cfg gives it the SAME R2FLight.ini settings the app uses,
    .myprint is a plain timestamped console print."""
    def __init__(self):
        self.cfg = R2FConfig.CFG()
        self.t0 = time.time()

    def myprint(self, text):
        print(f'{time.time() - self.t0:8.1f}  {text}')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--npoints', type=int, default=20)
    ap.add_argument('--points-per-switch', type=int, default=1)
    ap.add_argument('--outdir', default=r'C:\DATA\R2F\standalone')
    args = ap.parse_args()

    # Meas is a QObject (uses pyqtSignal.emit() internally) -- needs a Qt
    # application instance to exist, even though we never run an event loop
    QCoreApplication.instance() or QCoreApplication(sys.argv)

    parent = _MinimalParent()
    cfg = parent.cfg

    meas = Meas(QMutex(), parent, NDpts=1)

    # same V1/V2/fsig/modulation the GUI would use for the *current* ini
    # settings -- V2's amplitude isn't persisted to the ini (it's always
    # recomputed fresh by R2FLight.pyw's _compute_balance_v1v2() at Start),
    # so 6.0 V here matches that same standing convention
    fsig  = cfg.getfloatkey('FSIG')
    v1amp = cfg.getfloatkey('V1AMP')
    v1ang = cfg.getfloatkey('V1ANG')
    v2ang = cfg.getfloatkey('V2ANG')
    v2amp = 6.0
    modulation = not bool(cfg.getintkey('MODOFF'))
    V1c = v1amp * np.exp(1j * np.deg2rad(v1ang))
    V2c = v2amp * np.exp(1j * np.deg2rad(v2ang))
    meas.storeV(V1c, V2c, dV1=0.01, fsig=fsig, g1=1, g2=1, modulation=modulation)

    parent.myprint(f'fsig={fsig:g} Hz  sampcount={meas.sampcount}  fsamp={meas.fsamp}  '
                    f'V1={V1c.real:+.4f}{V1c.imag:+.4f}j  V2={V2c.real:+.4f}{V2c.imag:+.4f}j  '
                    f'modulation={modulation}')

    os.makedirs(args.outdir, exist_ok=True)
    out_path = os.path.join(args.outdir, 'sample2ch_' + time.strftime('%Y%m%d_%H%M%S') + '.npz')
    parent.myprint(f'writing to {out_path}')

    ch1_all, ch2_all, switch_all, ts_all = [], [], [], []

    try:
        for i in range(args.npoints):
            straight = (i // args.points_per_switch) % 2 == 0
            # drives prepForMeas()'s own relay-switch branch (self.co <
            # self.NDpts) -- NDpts=1, so co=0 -> straight, co=NDpts -> cross.
            # With NDpts=1, the modulation-ellipse angle inside prepForMeas
            # (self.co % self.NDpts) is always 0 either way, so V1 sits at a
            # fixed point rather than tracing an ellipse -- appropriate here
            # since this tool isn't doing the regression fit that needs it.
            meas.co = 0 if straight else meas.NDpts
            meas.prepForMeas()

            meas._wait(meas.settle_time)
            meas.sendtrig()
            acq_time = meas.sampcount / meas.fsamp
            meas._wait(max(1.0, acq_time * 1.2 + 0.3))

            # same channel-swap convention as Meas.getvals(): the relay
            # physically swaps which channel (101/102) sees which branch
            if meas.co < meas.NDpts:
                ch1 = meas.dvm.query_binary_values('FETCH3? (@101)', datatype='f', is_big_endian=True)
                ch2 = meas.dvm.query_binary_values('FETCH3? (@102)', datatype='f', is_big_endian=True)
            else:
                ch2 = meas.dvm.query_binary_values('FETCH3? (@101)', datatype='f', is_big_endian=True)
                ch1 = meas.dvm.query_binary_values('FETCH3? (@102)', datatype='f', is_big_endian=True)
            ch1 = np.asarray(ch1, dtype=float)
            ch2 = np.asarray(ch2, dtype=float)

            ch1_all.append(ch1)
            ch2_all.append(ch2)
            switch_all.append(0 if straight else 1)
            ts_all.append(time.time())

            parent.myprint(f'point {i + 1}/{args.npoints}  switch={"straight" if straight else "cross"}  '
                            f'ch1: mean={ch1.mean():+.6g} ptp={np.ptp(ch1):.6g}   '
                            f'ch2: mean={ch2.mean():+.6g} ptp={np.ptp(ch2):.6g}')

            # re-saved after every point so an interrupted run still leaves
            # a usable, complete file rather than nothing
            np.savez(out_path,
                     ch1=np.array(ch1_all), ch2=np.array(ch2_all),
                     switch=np.array(switch_all), ts=np.array(ts_all),
                     fsig=fsig, fsamp=meas.fsamp, sampcount=meas.sampcount)
    except KeyboardInterrupt:
        parent.myprint('Interrupted -- data collected so far is already saved')
    finally:
        meas._close_instruments()

    parent.myprint(f'Done -- saved {len(ch1_all)} points to {out_path}')


if __name__ == '__main__':
    main()
