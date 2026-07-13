import R2FLightAux
import numpy as np
from dataclasses import dataclass

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
    fit4: bool = True
    Cref: float = 100e-12
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
            'Yref' : 2*np.pi*self.cfg.fsig*self.cfg.Cref*1j,
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
            phi = np.angle(self.Data[i].Data[0].Vc)
            cf = np.exp(-1j*phi)
            for j in range(4):
                self.raw8[i,j] =  self.Data[i].Data[j].Vc*cf
            self.ctrl[i,0] =self.Data[i].V1c
            self.ctrl[i,1] =self.Data[i].V2c
        self.ave4  = 0.5*(self.raw8[::2,:]+self.raw8[1::2,:])
        self.ctrla = 0.5*(self.ctrl[::2,:]+self.ctrl[1::2,:])

        self.eta2 = self.ave4[:,1]/self.ave4[:,0]
        self.eta3 = self.ave4[:,2]/self.ave4[:,0]
        self.eta4 = self.ave4[:,3]/self.ave4[:,0]

    def calc(self):
        if self.cfg.modulation:
            self._calc_with_modulation()
        else:
            self._calc_no_modulation()
        self.setGoodFlag()

    def _calc_with_modulation(self):
        self.precalc()
        self.RawElli = np.zeros(self.N//2, dtype=object)
        self.EtaElli = np.zeros(3, dtype=object)
        for i in range(4):
            if i==3 and not self.cfg.fit4:
                break
            if i!=0:
                self.RawElli[i] = R2FLightAux.ComplexEllipse.fit_from_cmplx_points(self.ave4[:,i])
        self.EtaElli[0] = R2FLightAux.ComplexEllipse.fit_from_cmplx_points(self.eta2)
        self.EtaElli[1] = R2FLightAux.ComplexEllipse.fit_from_cmplx_points(self.eta3)
        if self.cfg.fit4:
            self.EtaElli[2] = R2FLightAux.ComplexEllipse.fit_from_cmplx_points(self.eta4)
        gain1_re = self.EtaElli[0].semi_major / self.EtaElli[1].semi_major
        gain1_im = self.EtaElli[0].semi_minor / self.EtaElli[1].semi_minor
        self.Res['mgain1']  = 0.5*(gain1_re + gain1_im)
        self.Res['mratio1'] = self.Res['mgain1']*self.EtaElli[1].eta_o - self.EtaElli[0].eta_o
        self.Res['R']    = np.real(1/(self.Res['Yref']*self.Res['mratio1']))
        self.Res['fnew'] = self.Res['fsig'] * -np.imag(self.Res['mratio1'])

    def _calc_no_modulation(self):
        """Estimate gain and R from noise correlation when V2 modulation is off.

        With no intentional modulation the N/2 phasors (eta2, eta3) cluster
        around a single point.  Their fluctuations are dominated by noise that
        is largely shared (common 1/V1 denominator noise), so the OLS regression
            d_eta2 = mgain * d_eta3  +  independent noise
        recovers the gain ratio robustly even with as few as 4 points.
        """
        self.precalc()
        # No ellipse objects — set to None so callers can guard against it
        self.RawElli = np.array([None] * (self.N//2))
        self.EtaElli = np.array([None, None, None])

        eta2_mean = np.mean(self.eta2)
        eta3_mean = np.mean(self.eta3)
        d_eta2 = self.eta2 - eta2_mean
        d_eta3 = self.eta3 - eta3_mean

        # OLS: d_eta2 = mgain * d_eta3  →  mgain = Σ(d_eta2·d_eta3*) / Σ|d_eta3|²
        denom = np.sum(np.abs(d_eta3)**2)
        if denom < 1e-30:
            # Essentially no signal variation — can't estimate gain
            self.goodData = False
            self.Res['mgain1']  = 0.0
            self.Res['mratio1'] = 0.0
            self.Res['R']       = float('nan')
            self.Res['fnew']    = self.Res['fsig']
            return
        gain_cplx = np.dot(d_eta2, np.conj(d_eta3)) / denom
        mgain = np.real(gain_cplx)   # gain is a real amplitude ratio

        self.Res['mgain1']  = mgain
        self.Res['mratio1'] = mgain * eta3_mean - eta2_mean
        self.Res['R']    = np.real(1/(self.Res['Yref']*self.Res['mratio1']))
        self.Res['fnew'] = self.Res['fsig'] * -np.imag(self.Res['mratio1'])

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
