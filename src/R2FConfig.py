import configparser as cp
import os

class CFG:
    k1= 'STD'
    std = { 'TZAGAIN': 4,
            'SAMPCOUNT': 300000,
            'NHARS': 9,
            'NPTS': 8,
            'MODOFF': 0,
            'FSIG': 1591.511,
            'SETTLE': 0.1,
            'SAVERAW': 0,
            'V2ANG': 90.0,
            'V1AMP': 5.9,
            'V1ANG': 0.0,
            'AUTOV1': 0,
            'AUTOTZA': 1,
            'REFPRESET': '100 pF',
            'DUTPRESET': '1 MOhm'}

    def __init__(self):
        self.fname = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'R2FLight.ini')
        self.cp = cp.ConfigParser()
        self.cp.read(self.fname)
    
    def save(self):
        with open( self.fname, 'w') as configfile:
            self.cp.write(configfile)            

    def getintkey(self,k2):
        if CFG.k1 in self.cp:
            if k2 in self.cp[CFG.k1]:
                return int(self.cp[CFG.k1][k2])
        if CFG.k1 not in self.cp:
            self.cp[CFG.k1]={}
        self.cp[CFG.k1][k2]='{0}'.format(CFG.std[k2])
        self.save()
        return int(CFG.std[k2])

    def setintkey(self,k2,val):
        if CFG.k1 not in self.cp:
            self.cp[CFG.k1]={}
        self.cp[CFG.k1][k2]='{0}'.format(val)
        self.save()
        return

    def getfloatkey(self,k2):
        if CFG.k1 in self.cp:
            if k2 in self.cp[CFG.k1]:
                return float(self.cp[CFG.k1][k2])
        if CFG.k1 not in self.cp:
            self.cp[CFG.k1]={}
        self.cp[CFG.k1][k2]='{0}'.format(CFG.std[k2])
        self.save()
        return float(CFG.std[k2])

    def setfloatkey(self,k2,val):
        if CFG.k1 not in self.cp:
            self.cp[CFG.k1]={}
        self.cp[CFG.k1][k2]='{0}'.format(val)
        self.save()
        return

    def getstrkey(self,k2):
        if CFG.k1 in self.cp:
            if k2 in self.cp[CFG.k1]:
                return self.cp[CFG.k1][k2]
        if CFG.k1 not in self.cp:
            self.cp[CFG.k1]={}
        self.cp[CFG.k1][k2]='{0}'.format(CFG.std[k2])
        self.save()
        return CFG.std[k2]

    def setstrkey(self,k2,val):
        if CFG.k1 not in self.cp:
            self.cp[CFG.k1]={}
        self.cp[CFG.k1][k2]='{0}'.format(val)
        self.save()
        return

