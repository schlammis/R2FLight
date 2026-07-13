import configparser as cp
import datetime
import os

class CFG:
    k1= 'STD'
    std = { 'SAMPLER': 'MY59012404',\
            'SIGGEN1':  'MY62004742',\
            'FS': 800000,\
            'LEN': 80000,\
            'F0': 1000,\
            'C2THR': 1E-6,\
            'V1': 0.11,\
            'V2': 0.11,\
            'PHASE2': 90.00,\
            'DATADIR': r'U:\002 - RFC\DATA',\
            'TZAGAIN': 4}

    def __init__(self):
        self.fname = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'R2F.ini')
        self.cp = cp.ConfigParser()
        self.cp.read(self.fname)
    
    def save(self):
        with open( self.fname, 'w') as configfile:
            self.cp.write(configfile)            

    def getstrkey(self,k2):
        if CFG.k1 in self.cp:
            if k2 in self.cp[CFG.k1]:
                return self.cp[CFG.k1][k2]
        if CFG.k1 not in self.cp:
             self.cp[CFG.k1]={}
        self.cp[CFG.k1][k2]='{0}'.format(CFG.std[k2])
        self.save()
        return CFG.std[k2]
    
    def getintkey(self,k2):
        if CFG.k1 in self.cp:
            if k2 in self.cp[CFG.k1]:
                return int(self.cp[CFG.k1][k2])
        if CFG.k1 not in self.cp:
            self.cp[CFG.k1]={}
        self.cp[CFG.k1][k2]='{0}'.format(CFG.std[k2])
        self.save()
        return int(CFG.std[k2])
        
    def getfloatkey(self,k2,fmt='{0:16.7f}'):
        if CFG.k1 in self.cp:
            if k2 in self.cp[CFG.k1]:
                return float(self.cp[CFG.k1][k2])
        if CFG.k1 not in self.cp:
            self.cp[CFG.k1]={}
        self.cp[CFG.k1][k2]=fmt.format(CFG.std[k2])
        self.save()
        return float(CFG.std[k2])

    def setfloatkey(self,k2,val,fmt='{0:16.7f}'):
        if CFG.k1 not in self.cp:
            self.cp[CFG.k1]={}
        self.cp[CFG.k1][k2]=fmt.format(val)
        self.save()
        return

    def setintkey(self,k2,val):
        if CFG.k1 not in self.cp:
            self.cp[CFG.k1]={}
        self.cp[CFG.k1][k2]='{0}'.format(val)
        self.save()
        return

