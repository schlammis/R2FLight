
from PyQt5.QtCore import (
QObject, 
pyqtSignal,
pyqtSlot,
QTimer)

import time
import numpy as np 
import pyvisa
import R2FMath
import CustomData
import re

class Meas(QObject):
    finished = pyqtSignal()
    dataReady  = pyqtSignal(CustomData.NPoints)
    dataSetReady  = pyqtSignal(CustomData.FourChannels)

    def __init__(self,mutex,parent,NDpts):
        super(QObject, self).__init__()
        self.NDpts = NDpts  #4 for a circle double points means in both switch positions
        self.Npts = 2*NDpts #8 for a circle double points means in both switch positions
        self.par =parent
        self.isidle =True
        self.mutex       = mutex
        self.co=-1
        self.Nhars=2
        self.runt = time.time()
        self.fsig = 1000
        self.fsamp = 800000 
        self.maxco=1200*2
        self._stop_requested = False
        self.modulation = True
        self.rm = pyvisa.ResourceManager()
        sg1_pattern ="^USB.*:MY62.*"
        dvm_pattern ="^USB.*:MY59.*"
        sg1_name=''
        dvm_name=''
        for i in self.rm.list_resources():
            if re.search(sg1_pattern,i):
                sg1_name=i
            elif re.search(dvm_pattern,i):   
                dvm_name=i
        self.sg1 = self.rm.open_resource(sg1_name)
        self.dvm = self.rm.open_resource(dvm_name)
        self.oldf=-1
        self.precmd()       

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
        #self.dvm.write('ACQ3:VOLT 3,(@101:104)')
        self.dvm.write('ACQ3:VOLT 3,DIFF,AC,TIME,(@101:102)')
        self.dvm.write('ACQ3:VOLT 0.3,DIFF,AC,TIME,(@103:104)')
        #self.dvm.write('ACQ3:VOLT 1,(@101)')
        self.dvm.write('SAMP3:RATE {0:8.2f},(@101:104)'.format(self.fsamp))
        self.dvm.write('SAMP3:COUN {0},(@101:104)'.format(300000))
        self.dvm.write('INP3:COUP AC,(@101:104)')
        self.dvm.write('TRIG3:SOUR BUS,(@101:104)')

    def write1dbg(self,ostr,debug=False):
        self.sg1.write(ostr)
        if debug:
            print(ostr)
        time.sleep(0.001)
 
    def storeV(self,V1c,V2c,dV2,fsig,g1,g2,modulation=True):
        if self.isidle:
            self.V1c=V1c
            self.V2c=V2c
            self.dV2 = dV2
            self.fsig = fsig
            self.g1 = g1
            self.g2 = g2
            self.modulation = modulation
        
    def prepForMeas(self):
        if self.co==0:
            self.par.myprint("V1= {0:8.3f}  V2={1:8.3f} dV2={2:8.3f}  f={3:8.5f} Hz".format(self.V1c,self.V2c,self.dV2,self.fsig/1000))
        if self.co%2==0:
            self.dvm.write('ROUT:OPEN (@211,248)')   
            self.dvm.write('ROUT:CLOS (@218,241)') # 8->1 1->4
        else:
            self.dvm.write('ROUT:OPEN (@218,241)')   
            self.dvm.write('ROUT:CLOS (@211,248)') # 1->1 8->4
        if self.modulation:
            ang = (self.co//2)/self.NDpts*2*np.pi
            a = self.dV2*1.2
            b = self.dV2*0.6
            theta = 30/180*np.pi
            self.V2 = self.V2c+np.exp(1j*theta)*(a*np.cos(ang)+1j*b*np.sin(ang))
        else:
            self.V2 = self.V2c
        self.V1 = self.V1c
        V1amp   = np.abs(self.V1)
        V2amp   = np.abs(self.V2)
#        if V1amp>=10.0:
#            print('Error V1>10 V -- rescaling')
#            self.V1 = 9/np.abs(self.V1)*self.V1
#            V1amp   = np.abs(self.V1)
#        elif V2amp>10.0:
#            print('Error V2>10 V -- rescaling')
#            self.V2 = 9.9/np.abs(self.V2)*self.V2
#            V2amp   = np.abs(self.V2)
        V1phase = np.angle(self.V1)/np.pi*180
        V2phase = np.angle(self.V2)/np.pi*180
        self.write1dbg('SOUR1:VOLT {0:8.4f}'.format(V1amp))
        self.write1dbg('SOUR2:VOLT {0:8.4f}'.format(V2amp))
        self.write1dbg('SOUR1:FREQ {0:8.4f}'.format(self.fsig))
        self.write1dbg('SOUR2:FREQ {0:8.4f}'.format(self.fsig))
        self.write1dbg('PHAS:SYNC')
        self.write1dbg('SOUR1:PHASE {0:8.4f}'.format(V1phase))
        self.write1dbg('SOUR2:PHASE {0:8.4f}'.format(V2phase))
        #self.par.myprint(f"cmd: Amp:{V1amp:.3f} Phase:{V1phase:.3f}")
        V1=float(self.sg1.query('SOUR1:VOLT?'))
        V2=float(self.sg1.query('SOUR2:VOLT?'))
        phase1=float(self.sg1.query('SOUR1:PHASE?'))
        phase2=float(self.sg1.query('SOUR2:PHASE?'))
        self.V1rb = V1 * np.exp(1j*phase1/180*np.pi)
        self.V2rb = V2 * np.exp(1j*phase2/180*np.pi)
        ret=self.sg1.query('SYSTem:ERRor?')
        retval = int(ret.split(',')[0])
        if retval!=0:
            self.par.myprint(f'Error: {ret}')


        
    def sendtrig(self):
        self.dvm.write('INIT3 (@101:104)')# %%
        self.dvm.write('*TRG')
       
    
    @pyqtSlot()
    def start(self):
        self.isidle=False
        start = time.time()
        my_config = CustomData.NPointsConfig(
            fsig=self.fsig,
            fsamp=self.fsamp,
            g1=self.g1,
            g2=self.g2,
            N=self.Npts,
            modulation=self.modulation
        )        
        self._stop_requested = False
        self._daq_armed = False
        self.rawN = CustomData.NPoints(my_config)
        for self.co in range(self.Npts):
            if self._stop_requested:
                break
            self.prepForMeas()
            self.sendtrig()
            self._daq_armed = True
            t_end = time.time() + 1.0
            while time.time() < t_end:
                if self._stop_requested:
                    break
                time.sleep(0.05)
            if self._stop_requested:
                break
            self._daq_armed = False
            self.getvals()
        self.isidle = True
        if not self._stop_requested:
            self.rawN.calc()
            self.dataReady.emit(self.rawN)
        self.finished.emit()

    @pyqtSlot()
    def stop(self):
        self._stop_requested = True
        if self._daq_armed:
            try:
                self.dvm.write('ABOR3,(@101:104)')
                self.dvm.write('*CLS')   # flush output queue so next session starts clean
            except Exception:
                pass


    def readCh(self,ch):
        """ ch = 1,2,3,4 """
        if ch not in (1,2,3,4):
            print('Channel not valid, you sent: {0}'.format(ch))
        ostr ='FETCH3? (@10{0})'.format(ch)
        values = self.dvm.query_binary_values(ostr, \
                datatype='f', is_big_endian=True)
        return values  
      
    def getvals(self):
        if self.co%2==0:
            ch1=self.dvm.query_binary_values('FETCH3? (@101)',  datatype='f', is_big_endian=True)
            ch2=self.dvm.query_binary_values('FETCH3? (@102)',  datatype='f', is_big_endian=True)
        else:
            ch2=self.dvm.query_binary_values('FETCH3? (@101)',  datatype='f', is_big_endian=True)
            ch1=self.dvm.query_binary_values('FETCH3? (@102)',  datatype='f', is_big_endian=True)
        ch3=self.dvm.query_binary_values('FETCH3? (@103)',  datatype='f', is_big_endian=True)
        ch4=self.dvm.query_binary_values('FETCH3? (@104)',  datatype='f', is_big_endian=True)
        self.rawN.setPoint(self.co,ch1,ch2,ch3,ch4,self.V1rb,self.V2rb,time.time())
        self.dataSetReady.emit(self.rawN.Data[self.co])
