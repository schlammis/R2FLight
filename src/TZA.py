"""
Module for TZA transimpedance amplifier.
Single device, COM port auto-detected at construction time.
"""

import serial
import serial.tools.list_ports


def find_tza_port():
    """Scan all COM ports and return the first one that responds to $U."""
    for info in serial.tools.list_ports.comports():
        try:
            with serial.Serial(info.device, baudrate=115200, bytesize=8,
                               parity='N', stopbits=1, timeout=0.5) as ser:
                ser.write(b'$U')
                resp = ser.read_until(b'\r')
                if resp:
                    print(f'TZA found on {info.device}')
                    return info.device
        except (serial.SerialException, OSError):
            continue
    return None


class TZA:
    POLDICT  = {0: 'not inverted', 1: 'inverted'}
    BWDICT   = {1: '10 kHz', 2: '1 kHz', 3: '100 Hz', 4: '10 Hz'}
    GAINDICT = {1: 'x1', 2: 'x10', 3: 'x100', 4: 'x1,000', 5: 'x10,000', 6: 'x100,000'}
    GAINFAC  = {1: 1, 2: 10, 3: 100, 4: 1000, 5: 10000, 6: 100000}

    def __init__(self, comport=None):
        """
        comport: explicit port string e.g. 'COM4', or None to auto-detect.
        """
        if comport is None:
            comport = find_tza_port()
        if comport is None:
            raise RuntimeError('TZA: no device found on any COM port')
        self.comport = comport
        self._ser = None
        self._init_device()

    # ------------------------------------------------------------------
    # Port management — keep open only while actively communicating
    # ------------------------------------------------------------------

    def _open(self):
        if self._ser is None or not self._ser.is_open:
            self._ser = serial.Serial(
                self.comport, baudrate=115200, bytesize=8,
                parity='N', stopbits=1, timeout=1, xonxoff=0, rtscts=0)

    def _close(self):
        if self._ser and self._ser.is_open:
            self._ser.close()

    def _cmd(self, cmd):
        """Send a command (bytes) and return the response bytes."""
        if isinstance(cmd, str):
            cmd = cmd.encode('utf-8')
        self._ser.write(cmd)
        return self._ser.read_until(b'\r')

    # ------------------------------------------------------------------
    # Device init and settings
    # ------------------------------------------------------------------

    def _init_device(self):
        self._open()
        try:
            self._cmd(b'$U')   # start
            self._cmd(b'$N')   # set non-inverting
            self._cmd(b'B1')   # bandwidth: 10 kHz
            self._read_settings()
        finally:
            self._close()

    def _read_settings(self):
        """Read gain, bandwidth, polarity from device. Port must already be open."""
        resp = self._cmd(b'$F')
        self.pol   = int(chr(resp[1]))
        self.hpol  = self.POLDICT[self.pol]

        resp = self._cmd(b'V?')
        self.gain  = int(chr(resp[1]))
        self.hgain = self.GAINDICT[self.gain]

        resp = self._cmd(b'B?')
        self.bw    = int(chr(resp[1]))
        self.hbw   = self.BWDICT[self.bw]

    def get_settings(self):
        """Refresh and return a dict of current settings."""
        self._open()
        try:
            self._read_settings()
        finally:
            self._close()
        return {'gain': self.gain, 'hgain': self.hgain,
                'bw': self.bw,   'hbw':   self.hbw,
                'pol': self.pol, 'hpol':  self.hpol}

    # ------------------------------------------------------------------
    # Gain
    # ------------------------------------------------------------------

    def set_gain(self, gain):
        """Set gain by index 1–6."""
        if not (1 <= gain <= 6):
            raise ValueError(f'TZA gain must be 1–6, got {gain}')
        self._open()
        try:
            self._cmd(f'V{gain}')
            self._read_settings()
        finally:
            self._close()

    def set_gain_by_label(self, label):
        """Set gain by human-readable label e.g. 'x100'."""
        for k, v in self.GAINDICT.items():
            if v == label:
                self.set_gain(k)
                return
        raise ValueError(f'TZA: unknown gain label "{label}"')

    def set_gain_by_factor(self, factor):
        """Set gain by numeric factor e.g. 100."""
        for k, v in self.GAINFAC.items():
            if v == factor:
                self.set_gain(k)
                return
        raise ValueError(f'TZA: unknown gain factor {factor}')

    def label_to_factor(self, label):
        for k, v in self.GAINDICT.items():
            if v == label:
                return self.GAINFAC[k]
        raise ValueError(f'TZA: unknown gain label "{label}"')

    # ------------------------------------------------------------------
    # Bandwidth
    # ------------------------------------------------------------------

    def set_bw(self, bw):
        """Set bandwidth by index 1–4."""
        if not (1 <= bw <= 4):
            raise ValueError(f'TZA bandwidth must be 1–4, got {bw}')
        self._open()
        try:
            self._cmd(f'B{bw}')
            self._read_settings()
        finally:
            self._close()

    # ------------------------------------------------------------------
    # Status string
    # ------------------------------------------------------------------

    def __str__(self):
        return (f'TZA on {self.comport} | '
                f'gain={self.hgain} | bw={self.hbw} | pol={self.hpol}')
