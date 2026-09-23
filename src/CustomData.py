import R2FLightAux
import numpy as np
from dataclasses import dataclass

# Known REF/DUT components, keyed by the label shown in the UI and stored in
# the ini (REFPRESET/DUTPRESET) -- (impedance type 'R' or 'C', nominal value
# in Ohms or Farads). Add entries here to make a new component available in
# both the REF and DUT dropdowns.
IMPEDANCE_PRESETS = {
    '100 pF':  ('C', 100e-12),
    '10 pF':   ('C', 10e-12),
    '1 MOhm':  ('R', 1e6),
    '100 kOhm': ('R', 100e3),
}


def nominal_impedance(kind, value, fsig):
    """Complex impedance of a nominal R or C component at fsig -- 'kind' is
    'R' (value in Ohms) or 'C' (value in Farads), matching IMPEDANCE_PRESETS."""
    if kind == 'C':
        return 1.0 / (1j * 2*np.pi*fsig*value)
    return complex(value)


def snap_ratio_prefactor(ratio):
    """Snap a nominal complex impedance ratio (Z_DUT/Z_REF) to the nearest
    'clean' design value: sign * 10**n sitting on whichever axis (real or
    imaginary) the ratio is actually on -- e.g. 10, -1, 0.1j. Same-type
    (R:R or C:C) comparisons land on the real axis; mixed-type (R:C or C:R)
    comparisons land on the imaginary axis, since Z_C is purely imaginary
    and Z_R is purely real. This is what the ratio should be by design; the
    caller reports how far the actual measurement deviates from it."""
    mag = 10 ** round(np.log10(abs(ratio)))
    if abs(ratio.real) >= abs(ratio.imag):
        return complex(mag * np.sign(ratio.real), 0.0)
    return complex(0.0, mag * np.sign(ratio.imag))


class SampleData:
    def __init__(self,fsig,fsamp,data,Nhars=2):
        self.data  = np.array(data)
        self.fsig  = fsig
        self.fsamp = fsamp
        self.Nhars = Nhars

    def setfmin(self,fmin,fline):
        self.fmin = fmin
        self.fline = fline

    def findf(self):
        self.fmin, self.fline = R2FLightAux.get_f(self.data,self.fsamp,self.fsig,Nhars=self.Nhars)
        return self.fmin, self.fline

    def fit(self):
        self.Vc, self.fv,self.c2,self.rss = R2FLightAux.fit_sine_cplx(
            self.data,self.fsamp,self.fmin,self.fline,Nhars=self.Nhars)


class FourChannels:
    def __init__(self,fsig,fsamp,Nhars,ch1,ch2,ch3,ch4,V1c,V2c,i,ts):
        self.ts    = ts
        self.fsig  = fsig
        self.fsamp = fsamp
        self.i=i
        if ts<0:
            return
        self.Data =[]
        self.V1c=V1c
        self.V2c=V2c
        self.Data.append(SampleData(fsig,fsamp,ch1,Nhars))
        self.Data.append(SampleData(fsig,fsamp,ch2,Nhars))
        self.Data.append(SampleData(fsig,fsamp,ch3,Nhars))
        self.Data.append(SampleData(fsig,fsamp,ch4,Nhars))
        fmin, fline = self.Data[0].findf()
        for i in range(4):
            self.Data[i].setfmin(fmin,fline)
            self.Data[i].fit()

@dataclass
class NPointsConfig:
    fsig: float
    fsamp: float
    Nhars: int = 2
    g1: float = 1.0
    g2: float = 1.0
    ratio: float = 10.0
    N: int = 8
    ref_type: str = 'C'   # 'C' (reference is a capacitor) or 'R' (a resistor)
    ref_value: float = 100e-12   # Farads if ref_type=='C', Ohms if ref_type=='R'
    modulation: bool = True
    

class NPoints:
    def __init__(self, config: NPointsConfig):
        self.cfg = config
        self.N=self.cfg.N
        self.ats = -1*np.ones(self.N)
        self.Data = np.zeros(self.N,dtype=object)

        self.Res = {
            'fsig': self.cfg.fsig,
            'ratio': self.cfg.ratio,
            'fsamp': self.cfg.fsamp,
            'Nhars': self.cfg.Nhars,
            'gain1': self.cfg.g1,
            'gain2': self.cfg.g2,
            # reference admittance: j*omega*C for a capacitive reference, or
            # 1/R for a resistive one -- Z_DUT = mratio1/Yref either way,
            # since Yref is only ever used as a plain complex divisor
            'Yref' : (2*np.pi*self.cfg.fsig*self.cfg.ref_value*1j if self.cfg.ref_type == 'C'
                      else 1.0/self.cfg.ref_value),
            'ts': min(self.ats)
        }        
        self.Res['ts']= min(self.ats)      

    def setPoint(self,i,ch1,ch2,ch3,ch4,V1c,V2c,ts):
        self.V1c=V1c
        self.Data[i] = FourChannels(self.Res['fsig'],self.Res['fsamp'],self.Res['Nhars'],\
                                    ch1,ch2,ch3,ch4,V1c,V2c,i,ts)
        self.ats[i] =ts
        if min(self.ats)>0:
            self.Res['ts'] = np.mean(self.ats)

    def precalc(self):
        self.raw8 = np.zeros((self.N,4),dtype=complex)
        self.ctrl = np.zeros((self.N,2),dtype=complex)
        for i in range(self.N):
            # V2 (channel 1, index 1) is held constant, so it's the stable
            # phase reference -- deroting by V1's own (modulated) phase
            # instead would trivially collapse V1's column to purely real.
            phi = np.angle(self.Data[i].Data[1].Vc)
            cf = np.exp(-1j*phi)
            for j in range(4):
                self.raw8[i,j] =  self.Data[i].Data[j].Vc*cf
            self.ctrl[i,0] =self.Data[i].V1c
            self.ctrl[i,1] =self.Data[i].V2c
        # First half of the points is one full ellipse in switch position A,
        # second half is a full ellipse in switch position B (same angle
        # sequence in both halves).
        half = self.N // 2
        self.raw8A, self.raw8B = self.raw8[:half,:], self.raw8[half:,:]
        self.ctrlA, self.ctrlB = self.ctrl[:half,:], self.ctrl[half:,:]

        # Switch-averaged view -- kept only for the eta*.dat/V1.dat logging.
        # The R calculation and the scatter-tab display both use the
        # per-switch-state values below instead.
        self.ave4  = 0.5*(self.raw8A+self.raw8B)
        self.ctrla = 0.5*(self.ctrlA+self.ctrlB)
        # eta1 = V1'/V2', eta3 = V3'/V2' (normalized by the Z_REF-branch
        # channel, per the bridge derivation -- NOT by V1/ch1).
        self.eta1 = self.ave4[:,0]/self.ave4[:,1]
        self.eta3 = self.ave4[:,2]/self.ave4[:,1]

        # Per-switch-state ratios, for the independent regression fits.
        self.eta1A = self.raw8A[:,0]/self.raw8A[:,1]
        self.eta3A = self.raw8A[:,2]/self.raw8A[:,1]
        self.eta1B = self.raw8B[:,0]/self.raw8B[:,1]
        self.eta3B = self.raw8B[:,2]/self.raw8B[:,1]

    def calc(self):
        if self.cfg.modulation:
            self._calc_with_modulation()
        else:
            self._calc_no_modulation()
        self.setGoodFlag()

    def _set_impedance_results(self, Z):
        """DUT modeled as R parallel with a parasitic C_p:
        Z = R/(1+j*omega*tau), tau = R*C_p. Inverting: Y = 1/Z = 1/R + j*omega*C_p,
        so R = 1/Re(Y), C_p = Im(Y)/omega, tau = R*C_p = Im(Y)/(omega*Re(Y))."""
        self.Res['Z'] = Z
        self.Res['X'] = np.imag(Z)
        Y = 1/Z
        self.Res['Y'] = Y
        w = 2*np.pi*self.Res['fsig']
        self.Res['R']   = 1/np.real(Y)
        self.Res['Cp']  = np.imag(Y)/w
        self.Res['tau'] = self.Res['Cp']*self.Res['R']

    def _calc_with_modulation(self):
        self.precalc()
        # V1 (raw channel 0) ellipse, fit per switch state -- scatter-tab
        # display only, purely a visual check on the raw trajectory shape.
        # V1 carries the modulation now; V2 is held constant.
        self.V1ElliA = R2FLightAux.ComplexEllipse.fit_from_cmplx_points(self.raw8A[:,0])
        self.V1ElliB = R2FLightAux.ComplexEllipse.fit_from_cmplx_points(self.raw8B[:,0])

        # Z_DUT/mratio1 comes from a linear regression of eta1 on eta3 for
        # each switch state, then averaging the two results. (The scatter
        # tab computes its own fitted-eta1 overlay live from partial data,
        # using this same regression -- see R2FLightAux.fit_eta_regression.)
        mgain1A, mratio1A = R2FLightAux.fit_eta_regression(self.eta1A,self.eta3A)
        mgain1B, mratio1B = R2FLightAux.fit_eta_regression(self.eta1B,self.eta3B)

        self.Res['mratio1A'] = mratio1A
        self.Res['mratio1B'] = mratio1B
        self.Res['mgain1']  = 0.5*(mgain1A + mgain1B)
        self.Res['mratio1'] = 0.5*(mratio1A + mratio1B)
        # mratio1 = Yref*Z_DUT directly (see derivation), so Z_DUT is just
        # mratio1/Yref -- and it's genuinely complex now, not just R.
        self._set_impedance_results(self.Res['mratio1'] / self.Res['Yref'])

    def _calc_no_modulation(self):
        """Estimate gain and R from noise correlation when V1 modulation is off.

        With no intentional modulation the N/2 phasors (eta1, eta3) cluster
        around a single point.  Their fluctuations are dominated by noise that
        is largely shared (common 1/V2 denominator noise), so the OLS regression
            d_eta1 = mgain * d_eta3  +  independent noise
        recovers the gain ratio robustly even with as few as 4 points.
        """
        self.precalc()
        # No ellipse objects — set to None so callers can guard against it
        self.V1ElliA = None
        self.V1ElliB = None

        eta1_mean = np.mean(self.eta1)
        eta3_mean = np.mean(self.eta3)
        d_eta1 = self.eta1 - eta1_mean
        d_eta3 = self.eta3 - eta3_mean

        # OLS: d_eta1 = mgain * d_eta3  →  mgain = Σ(d_eta1·d_eta3*) / Σ|d_eta3|²
        denom = np.sum(np.abs(d_eta3)**2)
        if denom < 1e-30:
            # Essentially no signal variation — can't estimate gain
            self.goodData = False
            self.Res['mgain1']  = 0.0
            self.Res['mratio1'] = 0.0
            self._set_impedance_results(complex('nan'))
            return
        gain_cplx = np.dot(d_eta1, np.conj(d_eta3)) / denom
        mgain = np.real(gain_cplx)   # gain is a real amplitude ratio

        self.Res['mgain1']  = mgain
        self.Res['mratio1'] = mgain * eta3_mean - eta1_mean
        self._set_impedance_results(self.Res['mratio1'] / self.Res['Yref'])

    def setGoodFlag(self):
        self.goodData=True


class DAQBuffer:
    def __init__(self, totLines,cols=6):
        self.totLines = totLines
        self.cols =cols
        # Use standard 'complex' or 'np.complex128' 
        self.data = np.zeros((totLines, self.cols), dtype=np.float64) 
        self.idx = 0
        self.is_full = False

    def add_point(self, pt):
        # 1. Insert the new row at the current write pointer
        self.data[self.idx] = pt
        
        # 2. Advance the pointer
        self.idx += 1
        
        # 3. If we hit the end of the array, wrap back to the beginning
        if self.idx >= self.totLines:
            self.idx = 0
            self.is_full = True

    def get_ordered_data(self):
        """Returns the data in chronological order for plotting."""
        if not self.is_full:
            # If we haven't wrapped around yet, just return the filled part
            return self.data[:self.idx]
        else:
            # If we HAVE wrapped around, the oldest data is at self.idx.
            # np.roll shifts the array so the oldest data is back at row 0.
            return np.roll(self.data, -self.idx, axis=0)
