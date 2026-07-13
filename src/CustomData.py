import sys
sys.path.append(r'c:\python\CeramicCap3\src')
import R2FMath
import numpy as np
import copy
from dataclasses import dataclass

class MyData:
    def __init__(self, ave4,ts):
        self.ave4 = ave4
        self.ts = ts

    def __str__(self):
        return f"TS: {self.ts}"


class SampleData:
    def __init__(self,fsig,fsamp,data,Nhars=2):
        self.data  = np.array(data)
        self.fsig  = fsig
        self.fsamp = fsamp
        self.f0 = self.fsig/self.fsamp
        self.Nhars = Nhars

    def setfmin(self,fmin):
        self.fmin = fmin

    def findf(self):
        self.fmin = R2FMath.get_f(self.data,self.f0)
        return self.fmin
        
    def fit(self):
        self.Vc, self.fv,self.c2 = R2FMath.fit_sine_cplx(self.data,self.fmin,self.Nhars)


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
        fmin = self.Data[0].findf()
        for i in range(4):
            self.Data[i].setfmin(fmin)
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
                self.RawElli[i] = R2FMath.ComplexEllipse.fit_from_cmplx_points(self.ave4[:,i])
        self.EtaElli[0] = R2FMath.ComplexEllipse.fit_from_cmplx_points(self.eta2)
        self.EtaElli[1] = R2FMath.ComplexEllipse.fit_from_cmplx_points(self.eta3)
        if self.cfg.fit4:
            self.EtaElli[2] = R2FMath.ComplexEllipse.fit_from_cmplx_points(self.eta4)
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

class AllData():
    def __init__(self):
        self.mydict={}

    def append(self,ND:NPoints):
        f = ND.Res['fsig']
        if f not in self.mydict:
            self.mydict[f] = []
        self.mydict[f].append(ND)
    
    def deletekey(self,f):
        if f in self.mydict:
             del self.mydict[f]

    def count(self):
        return len(list(self.mydict))

    def countf(self,f):
        if f not in self.mydict:
            return 0
        else:
            return len(self.mydict[f])
    
    def getkeys(self,f,keys):
        retdict ={}
        L = len(self.mydict[f])
        if L==0:
            for k in keys:
                retdict[k]=np.array([])
                return retdict
        for k in keys:
            retdict[k]=np.empty(L,dtype=np.array(self.mydict[f][0]).dtype)
        for n,a in enumerate(self.mydict[f]):
            for k in keys:
                retdict[k][n] =a.Res[k]
        return retdict

    def getallkeys(self,f):
        k = list(self.mydict)
        return self.getkeys(f,k)
    
    def getdictf (self,keys):
        retdict={}
        if 'fsig' not in keys:
            keys.append('fsig')
        allf = list(self.mydict)
        if len(allf)==0:
            for k in keys:
                retdict[k]=np.array([])
                return retdict
        else:
            for k in keys:
                retdict[k]=[]
            for f in allf:
                odict = self.getkeys(f,keys)
                for k in keys:
                    for item in odict[k]:
                        retdict[k].append(item)
            for k in keys:
                retdict[k] =np.array(retdict[k])
            return retdict


    def getAveVolts(self,f,t0=0):
        L = len(self.mydict[f])
        Nrows = np.shape(self.mydict[f][0].ave4)[0]  
        Ncols = 2+2*np.shape(self.mydict[f][0].ave4)[1]   +2*np.shape(self.mydict[f][0].ctrla)[1]
        #ret = np.empty(Nrows*L,Ncols)
        ret =[]
        
        obj = self.mydict[f][-1]
        Nrows = np.shape(obj.ave4)[0]
        for j in range(Nrows):
            line  = np.hstack((f,obj.Res['ts']-t0))
            for k in range(np.shape(obj.ave4)[1]):
                line =np.hstack((line,obj.ave4[j,k].real,obj.ave4[j,k].imag))
            for k in range(np.shape(obj.ctrla)[1]):
                line =np.hstack((line,obj.ctrla[j,k].real,obj.ctrla[j,k].imag))
            ret.append(line)            
        ret = np.array(ret)
        return ret


    def getRawPhasors(self,f,t0=0):
        L = len(self.mydict[f])
        Nrows = np.shape(self.mydict[f][0].raw8)[0]  
        Ncols = 2+2*np.shape(self.mydict[f][0].raw8)[1]   +2*np.shape(self.mydict[f][0].ctrla)[1]
        #ret = np.empty(Nrows*L,Ncols)
        ret =[]
        
        obj = self.mydict[f][-1]
        Nrows = np.shape(obj.raw8)[0]
        for j in range(Nrows):
            line  = np.hstack((f,obj.Res['ts']-t0))
            for k in range(np.shape(obj.raw8)[1]):
                line =np.hstack((line,obj.raw8[j,k].real,obj.raw8[j,k].imag))
            for k in range(np.shape(obj.ctrla)[1]):
                line =np.hstack((line,obj.ctrla[j//2,k].real,obj.ctrla[j//2,k].imag))
            ret.append(line)            
        ret = np.array(ret)
        return ret



    def getabf(self):
        f=list(self.mydict)
        ret=[]
        for ff in f:
            dummy=[]
            for a in self.mydict[ff]:
                dummy.append(np.hstack( (a.alphamean3,a.betamean3,a.alphamean4,a.betamean4)))
            dummy = np.array(dummy)
            da = dummy[:,2]-dummy[:,0]
            db = dummy[:,3]-dummy[:,1]
            if len(da)>2:
                sa =np.std(da,ddof=1)
            else:
                sa=0
            if len(db)>2:
                sb =np.std(da,ddof=1)
            else:
                sb=0
            ret.append(np.hstack((np.mean(da),np.mean(db),sa,sb)))

        return np.array(f),np.array(ret)

        




    

