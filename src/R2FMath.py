import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.optimize
from dataclasses import dataclass

"""
$\hat{\chi}^2 = a +bx +cx^2$
Minimum: 
$ x =-\frac{b}{2c}$ has the value $a-\frac{b^2}{2c} + \frac{b^2}{4c} = a-\frac{b^2}{4c}$
At the Minimum the $\chi^2$ should be the $NDF=N-3$. So,
$\chi^2 =\frac{\hat{\chi}^2}{\sigma^2}=N-3$. Hence, $\sigma^2 = \frac{\hat{\chi}^2}{N-3}$
The uncertainty in the parameter is  when the unscaled $\chi^2$ goes up by 1. That means the scaled one goes up by $\sigma^2$
$cx^2 = \sigma^2$ => $x=\sigma/c$
"""

def fit_sine(y,rf):
    """
    rf = relative frequency = f0/fs.
    sin(w t)= sin (2*pi*f0 *i/fs)= sin(2*pi*i *f0/fs)
    A sin(wt + phi) = A*sin(wt)*cos(phi)+A*cos(wt)*sin(phi)=S*sin(wt)+C*cos(w) => C/S=A*sin(phi)/A*cos(phi)=tan(phi) => phi=atan(C/S) 

    If we write 
    y  = Re( A*Exp(i (wt +phi)) = Re( (A* cos( w+ phi) + A*i*sin(w + phi))
    Re(A)* cos(wt + phi) - Im(A)* sin(w*t+phi)
    Re => cos_coeff
    Im => -sin_coeff
                   
    """
    i = np.arange(len(y))
    wt= 2*np.pi*i*rf
    O = np.ones(len(y))
    C = np.cos(wt)
    S = np.sin(wt)
    X = np.vstack((O,C,S)).T
    fit_pars = np.linalg.solve(X.T @ X, X.T @ y)
    fit_vals = X @ fit_pars
    C2 = np.dot(y-fit_vals,y-fit_vals)
    return fit_pars[1],fit_pars[2],fit_vals,C2

def fit_sine_cplx(y,rf,useHann=True):
    """
    rf = relative frequency = f0/fs.
    sin(w t)= sin (2*pi*f0 *i/fs)= sin(2*pi*i *f0/fs)
    A sin(wt + phi) = A*sin(wt)*cos(phi)+A*cos(wt)*sin(phi)=S*sin(wt)+C*cos(w) => C/S=A*sin(phi)/A*cos(phi)=tan(phi) => phi=atan(C/S) 

    If we write 
    y  = Re( A*Exp(i (wt +phi)) = Re( (A* cos( w+ phi) + A*i*sin(w + phi))
    Re(A)* cos(wt + phi) - Im(A)* sin(w*t+phi)
    Re => cos_coeff
    Im => -sin_coeff
                   
    """
    i = np.arange(len(y))
    wt= 2*np.pi*i*rf
    O = np.ones(len(y))
    C = np.cos(wt)
    S = np.sin(wt)
    X = np.vstack((O,C,S)).T
    if useHann:
        #W_vec = np.blackman(len(y))
        W_vec = np.hanning(len(y))
        yw = y * W_vec
        Ow = O * W_vec
        Cw = C * W_vec
        Sw = S * W_vec
        Xw = np.vstack((Ow,Cw,Sw)).T
        fit_pars = np.linalg.solve(Xw.T @ Xw, Xw.T @ yw)
    else:
        fit_pars = np.linalg.solve(X.T @ X, X.T @ y)
    fit_vals = X @ fit_pars
    C2 = np.dot(y-fit_vals,y-fit_vals)
    NDF = len(y)-3
    errv = np.sqrt(C2/NDF)
    return fit_pars[1]-1j*fit_pars[2],fit_vals,errv

def get_f(y,rf):
    """
    y = a+b*f+cf^2 => min b +2*c*f=0 => f= -b/(2c)
    """
    N = len(y)
    _,_,_,C20 = fit_sine(y,rf)
    _,_,_,C2m = fit_sine(y,rf*N/(N+1))
    _,_,_,C2p = fit_sine(y,rf*N/(N-1))
    ff = [rf*N/(N+1),rf,rf*N/(N-1)]
    yy = [C2m,C20,C2p]
    pf = np.polyfit(ff,yy,2)
    minf = -pf[1]/2/pf[0]
    if minf>rf*N/(N+1) and minf<rf*N/(N-1):
        return minf
    else:
        return get_f(y,minf)


def calcI2(f, C42,R=50,V2=-9.9):
    iw = 1j*f*2*np.pi
    I2 = V2*iw*C42/(1+iw*C42*R)
    return I2

def calcV1(f, C41,C42,R=50,V2=-9.9):
    iw = 1j*f*2*np.pi
    I2 = V2*iw*C42/(1+iw*C42*R)
    V1=-I2*(R+1/(iw*C41))
    return V1

def calcI1(f,C41,V1,R=50):
    iw = 1j*f*2*np.pi
    I1 = iw*C41/(1+iw*R*C41)*V1
    return I1
def lp(f,f0=2e4,Q=0.5):
    ret = f0*f0/(f0*f0-f*f+f0/Q*1j*f)
    return ret

def newgainvalue2(f,C41):
    V1 = calcV1(f,C41,C41/10)
    dI1 = calcI1(f,C41,V1*1e-2,R=50)
    gain=10**np.round(np.log10(0.1/np.abs(dI1*1e3*lp(f))))
    if gain<1: gain=1
    if gain>100000: gain =100000
    return gain

def newgainvalue1(f,C31,C41):
    V1 = calcV1(f,C41,C41/10)
    dI1 = calcI1(f,C31,V1*1e-2,R=50)
    gain=10**np.round(np.log10(0.1/np.abs(dI1*1e3*lp(f))))
    if gain<1: gain=1
    if gain>100000: gain =100000
    return gain


def mycomplexfit(x,y):
    L = len(x)
    X = np.empty((L*2,4))
    Y = np.empty((L*2,1))
    X[::2,0]  =  np.ones(L)
    X[1::2,0] =  np.zeros(L)
    X[::2,1]  =  np.zeros(L)
    X[1::2,1] =  np.ones(L)
    X[::2,2]  =  np.real(x)
    X[1::2,2] =  np.imag(x)
    X[::2,3]  = -np.imag(x)
    X[1::2,3] =  np.real(x)
    Y[::2,0]  =  np.real(y)
    Y[1::2,0] =  np.imag(y)
    C = np.linalg.inv(X.T @ X)
    fit_pars = C @ X.T @ Y
    fit_vals = X @ fit_pars
    diff = Y - fit_vals
    C2 = (diff.T @ diff)[0, 0]
    NDF = L*2-4
    vi = C2/NDF
    fit_errs = np.sqrt(np.diag(X @ C @ X.T) * vi)
    Cov = C * vi
    # returns fp, fv,fe,C2,Cov
    return fit_pars[:,0], fit_vals[::2,0] + 1j*fit_vals[1::2,0], fit_errs[::2]+1j*fit_errs[1::2], C2, Cov


class anaFile():
    def __init__(self,bd,fn,startix,ratio=10,data=None):
        if data is None or len(data)==0:
            self.bd = bd
            self.fn = fn
            data = np.loadtxt(os.path.join(self.bd,self.fn),skiprows=1+startix)
        self.raw = data
        #bi =startix
        L = len(data)
        Lh = (L//2)*2
        ave = 0.5*(data[0+1:Lh:2,:]+data[0:Lh-1:2,:])
        self.t   = ave[:,0]
        self.V1m =  ave[:,1]+1j*ave[:,2]
        self.V2m =  ave[:,3]+1j*ave[:,4]
        self.V3m =  ave[:,5]+1j*ave[:,6]
        self.V4m =  ave[:,7]+1j*ave[:,8]
        self.V1c =  ave[:,9]+1j*ave[:,10]
        self.V2c =  ave[:,11]+1j*ave[:,12]       
        self.allf = ave[:,13]
        self.routeco = ave[:,14]
        self.f   = np.mean(ave[:,13])
        self.u =self.V1m/self.V2m+ratio
        self.v3 = self.V3m/self.V2m
        self.v4 = self.V4m/self.V2m
        self.fp3, self.fv3,self.fe3,self.C23,self.Cov3 = mycomplexfit(self.v3,self.u)
        self.fp4, self.fv4,self.fe4,self.C24,self.Cov4 = mycomplexfit(self.v4,self.u)
        self.gain3  = self.fp3[2]+1j*self.fp3[3]
        self.gain4  = self.fp4[2]+1j*self.fp4[3]
        self.alpha3 = np.real(-self.u+ self.gain3*self.v3)/ratio
        self.beta3  = np.imag(-self.u+ self.gain3*self.v3)/ratio
        self.alpha4 = np.real(-self.u+ self.gain4*self.v4)/ratio
        self.beta4  = np.imag(-self.u+ self.gain4*self.v4)/ratio
        self.alphamean3 = np.mean(self.alpha3)
        self.betamean3 = np.mean(self.beta3)
        self.alphamean4 = np.mean(self.alpha4)
        self.betamean4 = np.mean(self.beta4)
        rtN = np.sqrt(len(self.alpha3))
        self.alphaerr3 = np.std(self.alpha3,ddof=1)/rtN
        self.betaerr3 = np.std(self.beta3,ddof=1)/rtN
        self.alphaerr4 = np.std(self.alpha4,ddof=1)/rtN
        self.betaerr4 = np.std(self.beta4,ddof=1)/rtN



class FACapBridge():
    def __init__(self,bd,maxf=1e5,skip=0,f0=1000,ratio=10):
        self.f0=f0
        self. bd = bd
        self.da  = []
        files =  [f for f in os.listdir(bd) if f.count("_")==2] 
        file_paths_with_time = [(os.path.getctime(os.path.join(bd, filename)), filename) for filename in files]
        file_paths_with_time.sort()
        sorted_files = [filename for creation_time, filename in file_paths_with_time]
        for fn in sorted_files:
            try:
                myc = anaFile(bd,fn,skip,ratio=ratio)
                if myc.f>maxf:
                    continue
                self.da.append( myc)        
            except:
                print(fn,end='')
                print('.....problem',end='')
                print()
        self.t      = []
        self.V1     = []
        self.V2     = []
        self.V3     = []
        self.V4     = []
        self.tm     = []
        self.f      = []
        self.gain3m = []
        self.gain4m = []
        self.alpha3m =[]
        self.alpha4m =[]
        self.beta3m =[]
        self.beta4m =[]
        self.alpha3e =[]
        self.alpha4e =[]
        self.beta3e =[]
        self.beta4e =[]

        for d in self.da:
            self.t.append(d.t)
            self.V1.append(np.mean(d.V1m))
            self.V2.append(np.mean(d.V2m))
            self.V3.append(d.V3m)
            self.V4.append(d.V4m)
            self.tm.append(np.mean(d.t))
            self.gain3m.append(np.mean(d.gain3))
            self.gain4m.append(np.mean(d.gain4))
            self.f.append(np.mean(d.f))
            self.alpha3m.append(d.alphamean3)
            self.alpha4m.append(d.alphamean4)
            self.beta3m.append(d.betamean3)
            self.beta4m.append(d.betamean4)
            self.alpha3e.append(d.alphaerr3)
            self.alpha4e.append(d.alphaerr4)
            self.beta3e.append(d.betaerr3)
            self.beta4e.append(d.betaerr4)

        self.t = np.concatenate(self.t)
        self.V1 = np.array(self.V1)
        self.V2 = np.array(self.V2)
        self.V3 = np.concatenate(self.V3)
        self.V4 = np.concatenate(self.V4)
        self.tm = np.array(self.tm)
        self.f = np.array(self.f)
        self.gain3m = np.array(self.gain3m)
        self.gain4m = np.array(self.gain4m)
        self.alpha3m = np.array(self.alpha3m)
        self.alpha4m = np.array(self.alpha4m)
        self.beta3m = np.array(self.beta3m)
        self.beta4m = np.array(self.beta4m)

        self.alpha3e = np.array(self.alpha3e)
        self.alpha4e = np.array(self.alpha4e)
        self.beta3e = np.array(self.beta3e)
        self.beta4e = np.array(self.beta4e)


        self.dal = self.alpha4m-self.alpha3m
        self.dbe = self.beta4m-self.beta3m

        self.dale = np.sqrt(self.alpha4e**2+self.alpha3e**2)
        self.dbee = np.sqrt(self.beta4e**2+self.beta3e**2)


        self.a3 = 232*(self.f/1e6)**2
        self.a4r    = self.a3 + self.dal
        self.b4r    = -self.dbe
        self.a1k =[]
        self.ix0=-1
        for i in range(len(self.f)):
            if self.f[i]==self.f0:
                if self.ix0<0:
                    self.ix0=i
                self.a1k.append(self.a4r[i])
        self.a1km = np.mean(self.a1k)
        self.a4r1k = self.a4r - self.a1km

        self.resdict={}

        for n in range(len(self.f)):
            f = self.f[n]
            if f not in self.resdict:
                self.resdict[f]={'ia3':[],'ib3':[],'ia4':[],'ib4':[],'ia3e':[],'ib3e':[],'ia4e':[],'ib4e':[]}
            self.resdict[f]['ia3'].append(self.alpha3m[n])
            self.resdict[f]['ia3e'].append(self.alpha3e[n])
            self.resdict[f]['ia4'].append(self.alpha4m[n])
            self.resdict[f]['ia4e'].append(self.alpha4e[n])
            self.resdict[f]['ib3'].append(self.beta3m[n])
            self.resdict[f]['ib3e'].append(self.beta3e[n])
            self.resdict[f]['ib4'].append(self.beta4m[n])
            self.resdict[f]['ib4e'].append(self.beta4e[n])
        for f in list(self.resdict):
            for k in  list(self.resdict[f]):
                self.resdict[f][k] = np.array(self.resdict[f][k])
            self.resdict[f]['a3'] = np.sum(self.resdict[f]['ia3']/self.resdict[f]['ia3e']**2)/np.sum(1/self.resdict[f]['ia3e']**2)
            self.resdict[f]['a4'] = np.sum(self.resdict[f]['ia4']/self.resdict[f]['ia4e']**2)/np.sum(1/self.resdict[f]['ia4e']**2)
            self.resdict[f]['b3'] = np.sum(self.resdict[f]['ib3']/self.resdict[f]['ib3e']**2)/np.sum(1/self.resdict[f]['ib3e']**2)
            self.resdict[f]['b4'] = np.sum(self.resdict[f]['ib4']/self.resdict[f]['ib4e']**2)/np.sum(1/self.resdict[f]['ib4e']**2)
            self.resdict[f]['a3e'] =1/np.sqrt(np.sum(1/(self.resdict[f]['ia3e'])**2))
            self.resdict[f]['a4e'] =1/np.sqrt(np.sum(1/(self.resdict[f]['ia4e'])**2))
            self.resdict[f]['b3e'] =1/np.sqrt(np.sum(1/(self.resdict[f]['ib3e'])**2))
            self.resdict[f]['b4e'] =1/np.sqrt(np.sum(1/(self.resdict[f]['ib4e'])**2))
            self.resdict[f]['da']  =  self.resdict[f]['a4'] - self.resdict[f]['a3']
            self.resdict[f]['db']  =  self.resdict[f]['b4'] - self.resdict[f]['b3']
            self.resdict[f]['dae'] = np.sqrt(self.resdict[f]['a3e']**2+ self.resdict[f]['a4e']**2)
            self.resdict[f]['dbe'] = np.sqrt(self.resdict[f]['b3e']**2+ self.resdict[f]['b4e']**2)
        f0 = self.f0
        if f0 in list(self.resdict):
            for f in list(self.resdict):
                for k in  list(self.resdict[f]):
                    self.resdict[f]['dar']  =  self.resdict[f]['da'] - self.resdict[f0]['da']

    def getkey(self,k):
        a=[]
        b=[]
        for f in sorted(list(self.resdict)):
            a.append(f)
            b.append(self.resdict[f][k])
        return np.array(a),np.array(b)
    
    def darplot(self,mul=1):
        a=[]
        b=[]
        c=[]
        for f in sorted(list(self.resdict)):
            a.append(f)
            b.append(self.resdict[f]['dar']*mul)
            c.append(self.resdict[f]['dae']*mul)
        return np.array(a),np.array(b),np.array(c)


    def dbeplot(self,mul=1):
        a=[]
        b=[]
        c=[]
        for f in sorted(list(self.resdict)):
            a.append(f)
            b.append(self.resdict[f]['db']*mul)
            c.append(self.resdict[f]['dbe']*mul)
        return np.array(a),np.array(b),np.array(c)


class FourCplxPts():
    """
    Originally this function used 4 points, in reality it can be 4+ points
    """
    def __init__(self,data):
        self.data = data
        self.cplxcenter,self.cplxradius,self.C2 = self.fit_circle(np.real(data),np.imag(data))

    def plot_circle(self,par):
        ang = np.exp(1j*np.linspace(0,2*np.pi))
        return np.real(par[0]+par[1]*ang),np.imag(par[0]+par[1]*ang)
    
    def plot_mycircle(self):
        return self.plot_circle([self.cplxcenter,self.cplxradius])

    def fit_circle(self,x,y,sigma=1):
        # points: array-like shape (n,2), n >= 3
        A = np.c_[x, y, np.ones_like(x)]
        b = -(x**2 + y**2)
        D, E, F = np.linalg.lstsq(A, b, rcond=None)[0]
        x0, y0 = -D/2, -E/2
        c0 = x0 +1j*y0
        r = np.sqrt((D**2 + E**2)/4 - F)
        phi0 = np.arctan2(y[0]-y0,x[0]-x0)
        cr = r*np.exp(1j*phi0)
        d = np.sqrt((x-x0)**2 + (y-y0)**2)
        residuals = d - r
        chi2 = np.sum((residuals/sigma)**2)
        return c0, cr,chi2



class FourPlusCplxPts():
    """
    For the ellipse fitting, we need at least 6 non colinaear points.
    Fittign algorithm:
       A.~Fitzgibbon, M.~Pilu, and R.~B.~Fisher, 
       ``Direct least square fitting of ellipses,'' 
       \emph{IEEE Transactions on Pattern Analysis and Machine Intelligence}, vol.~21, no.~5, pp.~476--480, May 1999.
    """
    def __init__(self,data):
        self.data = data

        #print(self.fit_ellipse(np.real(data),np.imag(data)))
        self.cplxcenter,self.cplxradius,self.cplxangle,self.C2 = self.fit_ellipse(np.real(data),np.imag(data))

    def plot_circle(self,par):
        cplxcenter, cplxsemi,cplxangle =par
        angle = np.imag(cplxangle)
        t    =   np.linspace(0,2*np.pi,200)
        
        outx = np.real(cplxcenter) + np.real(cplxsemi)*np.cos(t)*np.cos(angle) - np.imag(cplxsemi)*np.sin(t)*np.sin(angle)
        outy = np.imag(cplxcenter) + np.real(cplxsemi)*np.cos(t)*np.sin(angle) + np.imag(cplxsemi)*np.sin(t)*np.cos(angle)
        col = t-np.real(cplxangle)
        return {'x':outx,'y':outy,'c':col}

    def plot_mycircle(self):
        return self.plot_circle([self.cplxcenter,self.cplxradius,self.cplxangle])


    def fit_ellipse(self,x, y):
        """
        Fits an ellipse to a set of points (x, y) using the Fitzgibbon Direct Least Squares method.
        Returns the geometric parameters: center, width, height, and angle.
        """
        x = np.array(x)
        y = np.array(y)
        # --- 1. Construct the Design Matrix D: Form: Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0
        D = np.vstack([x**2, x*y, y**2, x, y, np.ones_like(x)]).T       
        # --- 2. Construct the Scatter Matrix S ---
        S = np.dot(D.T, D)
        # --- 3. Construct the Constraint Matrix C ---
        # Constraint: 4AC - B^2 = 1
        C = np.zeros((6, 6))
        C[0, 2] = C[2, 0] = 2
        C[1, 1] = -1

        # --- 4. Solve Generalized Eigensystem ---
        # We want to solve S*a = lambda*C*a. 
        # Since C is singular, we solve inv(S)*C*a = (1/lambda)*a
        # This is equivalent to finding eigenvalues of inv(S) @ C
        
        try:
            # Note: In some robust implementations, you might use scipy.linalg.eig
            # but numpy.linalg.eig is usually sufficient for this scope.
            E, V = np.linalg.eig(np.dot(np.linalg.inv(S), C))
        except np.linalg.LinAlgError:
            print("Singular matrix encountered. Points may be collinear.")
            return None

        # --- 5. Find the Single Positive Eigenvalue ---
        # The solution corresponds to the eigenvector with the only positive eigenvalue
        n = np.argmax(E)
        a_vec = V[:, n]
        a,b,c,d,e,f = a_vec

        if np.abs(a)<1e-8:
            cx = np.mean(x)
            cy = np.mean(y)
            major=0
            minor=0
            theta = 0
        else:
            cx, cy, major, minor, theta = self.cart_to_pol(a_vec)


        # --- 6. Convert Algebraic Coefficients to Geometric Parameters ---
        # a_vec = [A, B, C, D, E, F]

     
        test_x = cx + minor * np.cos(theta)
        test_y = cy + minor * np.sin(theta)
        
        # 2. Calculate the algebraic residual for this point: Ax^2 + Bxy + ...
        # Note: We use the `a, b, c, d, e, f` coefficients from the fit
        residual_at_minor = (a * test_x**2 + b * test_x * test_y + c * test_y**2 + 
                            d * test_x + e * test_y + f)

        # 3. Compare with a test point at distance `major`
        test_x_maj = cx + major * np.cos(theta)
        test_y_maj = cy + major * np.sin(theta)
        residual_at_major = (a * test_x_maj**2 + b * test_x_maj * test_y_maj + c * test_y_maj**2 + 
                            d * test_x_maj + e * test_y_maj + f)
        
        # 4. Decide
        # If the point at distance `minor` has a lower error, then `theta`
        # was actually pointing to the Minor axis.
        if abs(residual_at_minor) < abs(residual_at_major):
            # We want theta to always point to the Major axis for consistency.
            # So we rotate it by 90 degrees.
            theta += np.pi / 2
        rpts = np.sqrt((x-cx)**2+(y-cy)**2)
        phi = np.arctan2(y-cy,x-cx)
        phi_rel = phi-theta
        numerator = major * minor
        denominator = np.sqrt((minor * np.cos(phi_rel))**2 + (major * np.sin(phi_rel))**2)
        rellipse = numerator / denominator
        residuals = rpts - rellipse
        c2 = np.sum(residuals**2)
        phi0 = np.arctan2(y[0]-cy,x[0]-cx)
 
        return    cx+1j*cy, major+1j*minor, phi0+1j*theta,c2

    def cart_to_pol(self,coeffs):
        """
        Converts algebraic ellipse coefficients to geometric parameters.
        """
        a, b, c, d, e, f = coeffs
        # The formulas below are derived from the general conic equation
        b2 = b**2
        a_minus_c = a - c       
        # Tolerance for circle case (b=0, a=c)
        if b == 0 and abs(a - c) < 1e-6:
            # It is a circle
            cx = -d / (2*a)
            cy = -e / (2*a)
            radius = np.sqrt((d**2 + e**2) / (4*a**2) - f/a)
            return cx, cy, radius, radius, 0.0
        # 1. Calculate the center (h, k)
        num = b2 - 4*a*c
        cx = (2*c*d - b*e) / num
        cy = (2*a*e - b*d) / num
        # 2. Calculate the Angle (theta)
        # The angle of the major axis relative to the x-axis
        # Note: different sources use different definitions (major vs minor).
        # This formula typically gives the angle of the ellipse's rotation.
        theta = 0.5 * np.arctan2(b, a_minus_c) + np.pi/2
        # 3. Calculate Semi-Axes lengths
        # Using the standard conversion formulas
        up = 2 * (a*e**2 + c*d**2 - b*d*e + (b2 - 4*a*c)*f)
        down1 = (b2 - 4*a*c) * ((a + c) + np.sqrt((a - c)**2 + b2))
        down2 = (b2 - 4*a*c) * ((a + c) - np.sqrt((a - c)**2 + b2))
        # We use abs() because the sign of the eigenvector is arbitrary
        res1 = np.sqrt(abs(up / down1))
        res2 = np.sqrt(abs(up / down2))
        # Major axis is the larger one
        major_axis = max(res1, res2)
        minor_axis = min(res1, res2)
        return cx, cy, major_axis, minor_axis, theta        





class Aset():
    def __init__(self,bd,dir1000pF,dir10nF,dir100nF,dir1uF,dir10uF,skip=2*4*5):
        self.C1000pF = FACapBridge(self.gd(bd,dir1000pF),skip=2*4*5)
        self.C10nF   = FACapBridge(self.gd(bd,dir10nF),skip=2*4*5)
        self.C100nF  = FACapBridge(self.gd(bd,dir100nF),skip=2*4*5)
        self.C1uF    = FACapBridge(self.gd(bd,dir1uF),skip=2*4*5)
        self.C10uF   = FACapBridge(self.gd(bd,dir10uF),skip=2*4*5)
        self.pref =1e6/np.array([1,10,100,1000,1e4])


    def calcdiff(self):
        self.f1000pF,self.Cap1000pF  = self.C1000pF.getkey('dar')
        _,self.Cape1000pF = self.C1000pF.getkey('dae')
        self.Cap1000pF = self.Cap1000pF-self.Cap1000pF
        
        self.f10nF,self.Cap10nF  = self.C10nF.getkey('dar')
        _,self.Cape10nF = self.C10nF.getkey('dae')
        self.Cap10nF  = self.Cap10nF +self.Cap1000pF

        self.f100nF,self.Cap100nF  = self.C100nF.getkey('dar')
        _,self.Cape100nF = self.C100nF.getkey('dae')
        self.Cap100nF  = self.Cap100nF +self.Cap10nF

        self.f1uF,self.Cap1uF  = self.C1uF.getkey('dar')
        _,self.Cape1uF = self.C1uF.getkey('dae')
        self.Cap1uF  = self.Cap1uF +self.Cap100nF

        self.f10uF,self.Cap10uF  = self.C10uF.getkey('dar')
        _,self.Cape10uF = self.C10uF.getkey('dae')
        self.Cap10uF  = self.Cap10uF +self.Cap1uF


        _,self.D1000pF  = self.C1000pF.getkey('db')
        self.D1000pF=self.D1000pF-self.D1000pF
        _,self.De1000pF = self.C1000pF.getkey('dbe')

        _,self.D10nF  = self.C10nF.getkey('db')
        self.D10nF=self.D1000pF-self.D10nF
        _,self.De10nF = self.C10nF.getkey('dbe')

        _,self.D100nF  = self.C100nF.getkey('db')
        self.D100nF=self.D10nF-self.D100nF
        _,self.De100nF = self.C100nF.getkey('dbe')

        _,self.D1uF  = self.C1uF.getkey('db')
        self.D1uF=self.D100nF-self.D1uF
        _,self.De1uF = self.C1uF.getkey('dbe')

        _,self.D10uF  = self.C10uF.getkey('db')
        self.D10uF=self.D1uF-self.D10uF
        _,self.De10uF = self.C10uF.getkey('dbe')



    def gd(self,bd,fn):
        return os.path.join(bd,fn[0:6],fn)
    

    def plotraw(self,fig=None,ax=[],co='ro'):
        mul=['x1e6','x1e5','x1e4','x1e3','x1e2']
        if fig is None:
            fig,((ax1,ax2),(ax3,ax4),(ax5,ax6),(ax7,ax8),(ax9,axA)) = plt.subplots(5,2,figsize=(8,10))
            fig.subplots_adjust(wspace=0.4,hspace=0)
            ax = [ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9,axA]

        ax[0].errorbar(*self.C1000pF.darplot(self.pref[0]),fmt=co,label='1000pF/100pF')
        ax[2].errorbar(*self.C10nF.darplot(self.pref[1]),fmt=co,label='10nF/1000pF')
        ax[4].errorbar(*self.C100nF.darplot(self.pref[2]),fmt=co,label='100nF/10nF')
        ax[6].errorbar(*self.C1uF.darplot(self.pref[3]),fmt=co,label='1uF/100nF')
        ax[8].errorbar(*self.C10uF.darplot(self.pref[4]),fmt=co,label='10uF/1uF')

        ax[1].errorbar(*self.C1000pF.dbeplot(self.pref[0]),fmt=co,label='1000pF/100pF')
        ax[3].errorbar(*self.C10nF.dbeplot(self.pref[1]),fmt=co,label='10nF/1000pF')
        ax[5].errorbar(*self.C100nF.dbeplot(self.pref[2]),fmt=co,label='100nF/10nF')
        ax[7].errorbar(*self.C1uF.dbeplot(self.pref[3]),fmt=co,label='1uF/100nF')
        ax[9].errorbar(*self.C10uF.dbeplot(self.pref[4]),fmt=co,label='10uF/1uF')


        for n,a in enumerate(ax):
            a.set_xscale('log')
            a.set_xlim(80,1.1e5)
            if n<8:
                a.xaxis.set_ticklabels([])
                a.tick_params(axis='x',direction='inout')

            if n%2==0:
                a.set_ylabel('$\Delta$ alpha '+mul[n//2])
                a.legend()
            else:
                a.set_ylabel('$\Delta$ beta '+mul[n//2])
        return fig,ax


    def plotdiff(self,fig=None,ax=[],co='ro'):
        pref =1e6/np.array([1,10,100,1000,1e4])
        mul=['x1e6','x1e5','x1e4','x1e3','x1e2']
        if fig is None:
            fig,((ax1,ax2),(ax3,ax4),(ax5,ax6),(ax7,ax8)) = plt.subplots(4,2,figsize=(8,10))
            fig.subplots_adjust(wspace=0.4,hspace=0)
            ax = [ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8]
        ax[0].errorbar(self.f10nF  ,self.Cap10nF  *pref[1],self.Cape10nF  *pref[1],fmt=co,label='10nF')
        ax[2].errorbar(self.f100nF ,self.Cap100nF *pref[2],self.Cape100nF *pref[2],fmt=co,label='100nF')
        ax[4].errorbar(self.f1uF   ,self.Cap1uF   *pref[3],self.Cape1uF   *pref[3],fmt=co,label='1uF')
        ax[6].errorbar(self.f10uF  ,self.Cap10uF  *pref[4],self.Cape10uF  *pref[4],fmt=co,label='10uF')


        ax[1].errorbar(self.f10nF  ,self.D10nF  *pref[1],self.De10nF  *pref[1],fmt=co,label='10nF')
        ax[3].errorbar(self.f100nF ,self.D100nF *pref[2],self.De100nF *pref[2],fmt=co,label='100nF')
        ax[5].errorbar(self.f1uF   ,self.D1uF   *pref[3],self.De1uF   *pref[3],fmt=co,label='1uF')
        ax[7].errorbar(self.f10uF  ,self.D10uF  *pref[4],self.De10uF  *pref[4],fmt=co,label='10uF')

        for n,a in enumerate(ax):
            a.set_xscale('log')
            a.set_xlim(80,1.1e5)
            if n<6:
                a.xaxis.set_ticklabels([])
                a.tick_params(axis='x',direction='inout')

            if n%2==0:
                a.set_ylabel('$\Delta$ C/C(f0) '+mul[n//2])
                a.legend()
            else:
                a.set_ylabel('D(f) '+mul[n//2])
        return fig,ax

@dataclass
class ComplexEllipse:
    """
    Represents an ellipse parameterized by counter-rotating complex amplitudes.
    Optimized for R2F converter signal processing.
    """
    eta_o: complex
    eta_ccw: complex
    eta_cw: complex

    @property
    def center(self) -> tuple[float, float]:
        """Returns the geometric center (cx, cy)."""
        return np.real(self.eta_o), np.imag(self.eta_o)

    @property
    def semi_major(self) -> float:
        """Returns the semi-major axis length."""
        return complex(self.eta_ccw + self.eta_cw)

    @property
    def semi_minor(self) -> float:
        """Returns the semi-minor axis length."""
        return complex((self.eta_ccw - self.eta_cw)*np.exp(1j*np.pi/2))

    @property
    def angle(self) -> float:
        """Returns the rotation angle in radians."""
        return float(np.angle(self.eta_ccw + self.eta_cw))

    def evaluate(self, N: int = 100) -> np.ndarray:
        """Generates the complex trajectory for N points."""
        n = np.arange(N)
        phase = 2 * np.pi * n / N
        return self.eta_o + self.eta_ccw * np.exp(1j * phase) + self.eta_cw * np.exp(-1j * phase)

    @classmethod
    def fit_from_points(cls, x: np.ndarray, y: np.ndarray):
        """
        Fits an ellipse to (x,y) data using the Fitzgibbon Direct Least Squares method
        and returns a pure ComplexEllipse instance.
        """
        
        
        x = np.array(x)
        y = np.array(y)

        mx, my = np.mean(x), np.mean(y)
        scale = np.max([np.std(x), np.std(y)]) 
        if scale == 0: 
            return None
        
        xn = (x - mx) / scale
        yn = (y - my) / scale

        # 1. Construct matrices
        D = np.vstack([xn**2, xn*yn, yn**2, xn, yn, np.ones_like(xn)]).T       
        S = np.dot(D.T, D)
        
        C = np.zeros((6, 6))
        C[0, 2] = C[2, 0] = 2
        C[1, 1] = -1

        # 2. Solve the eigensystem
        try:
            epsilon = 1e-10  # Regularization for noiseless edge cases
            S = S + epsilon * np.eye(6)
            E, V = np.linalg.eig(np.dot(np.linalg.inv(S), C))
        except np.linalg.LinAlgError:
            return None

        # 3. Extract algebraic coefficients
        n = np.argmax(E)
        A, B, C_coeff, D_coeff, E_coeff, F = V[:, n]

        # 4. Convert Algebraic to Geometric Parameters
        # Center coordinates
        denominator = B**2 - 4 * A * C_coeff
        if denominator >= 0:
            return None # Not an ellipse
            
        cx = (2 * C_coeff * D_coeff - B * E_coeff) / denominator
        cy = (2 * A * E_coeff - B * D_coeff) / denominator

        # Semi-axes lengths
        numerator = 2 * (A * E_coeff**2 + C_coeff * D_coeff**2 - B * D_coeff * E_coeff + denominator * F)
        term2 = np.sqrt((A - C_coeff)**2 + B**2)
        
        axis1 = np.sqrt(abs(numerator / (denominator * (A + C_coeff + term2))))
        axis2 = np.sqrt(abs(numerator / (denominator * (A + C_coeff - term2))))
        
        major = max(axis1, axis2)
        minor = min(axis1, axis2)
        # Rotation angle
        theta = 0.5 * np.arctan2(B, A - C_coeff)
        phi = np.arctan2(y-cy,x-cx)        
        x_unrotated = major * np.cos(phi)
        y_unrotated = minor * np.sin(phi)
        test1_x = x_unrotated * np.cos(theta) - y_unrotated * np.sin(theta) + cx
        test1_y = x_unrotated * np.sin(theta) + y_unrotated * np.cos(theta) + cy

        residual1  = (A * test1_x**2 + B * test1_x * test1_y + C_coeff * test1_y**2 + 
                             D_coeff * test1_x + E_coeff * test1_y + F)
        res1= np.dot(residual1 ,residual1 )
        test2_x = x_unrotated * np.cos(theta+np.pi/2) - y_unrotated * np.sin(theta+np.pi/2) + cx
        test2_y = x_unrotated * np.sin(theta+np.pi/2) + y_unrotated * np.cos(theta+np.pi/2) + cy

        residual2  = (A * test2_x**2 + B * test2_x * test2_y + C_coeff * test2_y**2 + 
                             D_coeff * test2_x + E_coeff * test2_y + F)
        res2= np.dot(residual2 ,residual2 )

        if res2<res1:
            theta+=np.pi/2
            c2=res2
        else:
            c2=res1

        phi0 = np.arctan2(yn[0]-cy,xn[0]-cx)
        #print(phi0,theta)
        

        # 5. Calculate Counter-Rotating Complex Amplitudes
        eta_o = cx*scale+mx + 1j *(( cy*scale) +my)
        #phase_factor = np.exp(1j * theta)

        a = np.exp(1j*phi0)
        b = np.exp(1j*theta)
        c = np.exp(1j*(theta+np.pi))
        if abs(np.angle(c/a))<np.abs(np.angle(b/a)):
            theta = theta+np.pi
        

        #phase_factor = np.exp(1j * phi0)
        phase_factor = np.exp(1j * theta)
        eta_ccw = 0.5 * (major + minor) * phase_factor *scale
        eta_cw = 0.5 * (major - minor) * phase_factor *scale

        return cls(eta_o=eta_o, eta_ccw=eta_ccw, eta_cw=eta_cw)
    
    @classmethod
    def fit_from_cmplx_points(cls, cplx):
        return cls.fit_from_points(np.real(cplx),np.imag(cplx))

    
    # Inside your ComplexEllipse dataclass in R2FMath.py

    def simulate_noisy_data(self, N: int, noise_std: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        """
        Generates N points along the ellipse with optional Gaussian noise.
        Returns a tuple of (x_array, y_array) to be used for testing.
        """
        # 1. Get the perfect mathematical path
        z_perfect = self.evaluate(N)
        
        # 2. Add random Gaussian noise to the real and imaginary parts
        x = np.real(z_perfect) + np.random.normal(0, noise_std, N)
        y = np.imag(z_perfect) + np.random.normal(0, noise_std, N)
        
        return x, y
    
    def plot_elli(self,ax,ellipse_color='k',maj_color='r',min_color='b',rescale=True):
        data = self.evaluate()
        ax.plot(np.real(data),np.imag(data),linestyle='-',color=ellipse_color)
        l1 = np.array([self.eta_o,self.eta_o+self.semi_major])
        l2 = np.array([self.eta_o,self.eta_o+self.semi_minor])
        ax.plot(np.real(l1),np.imag(l1),linestyle='-.',color= maj_color)
        ax.plot(np.real(l2),np.imag(l2),linestyle=':',color=min_color)
        if rescale:
            bex,enx = ax.get_xlim()
            bey,eny = ax.get_ylim()
            delx,mex = 0.5*(enx-bex), 0.5*(enx+bex)
            dely,mey = 0.5*(eny-bey), 0.5*(eny+bey)
            if delx>dely: 
                delm=delx
            else:
                delm=dely

            bex,enx = mex-delm ,mex+delm
            bey,eny = mey-delm ,mey+delm
            ax.set_xlim(bex,enx)
            ax.set_ylim(bey,eny)

      
