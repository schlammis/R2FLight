#!/usr/bin/env python
""" A module to compute power spectra,
cross spectral densities, coherence, &
response function

TODO: Need to check if normalization also works if
the function is called

Note: Average the stuff in the functions, csd, psd, ..
PyScripter
ipython


"""


__author__          =   "Stephan Schlamminger"
__email__           =   "schlammi@gmail.com"
__status__          =   "Development"
__date__            =   "11/20/11"
__version__         =   "0.1"


import math
import numpy as np


def f_hann(x,N):
    """
    Calculates the hann window for a window of length N at the point x  elem 0..N-1
    """
    h=0.0
    if (x>=0) and (x<=N-1):
        h=1.0-math.cos(2*math.pi*x/(N-1.0))
    return h


def win_hann(winlen):
    x=range(winlen)
    h=[f_hann(i,winlen) for i in x]
    return h

def win_rect(winlen):
    x=np.ones(winlen)
    return list(x)


def calc_win(winlen,wintype=2):
    if wintype==2:
        mywindow=win_hann(winlen)
    else:
        mywindow=win_rect(winlen)
    return mywindow

def norm_win(window):
    s=0;
    t=0;
    for i in window:
        s+=i*i
        t+=1
    return 1.0*s/t


def get_segment_starts(N,L,D):
    """
        This function returns the startindices for the overlapping segements to
        calculate the PSD. The list looks like [0,ix1,ix2,ix3,..]. The length
        of the original data stream is N. The length of one segment is L and D
        is the overlap. The  segments run from 0..L-1,ix1..ix1+L-1,....,ixK..
        ixK+L-1, with ix1=D,ix2=2*D,ix3=3*D...
        Note this is in python notation data[ix1:ix1+L]
    """
    a=[0]
    ss=0
    if D==0: return a
    while ss+D+L<=N:
        ss=ss+D
        a.append(ss)
    return a


def mypsd(ind1,dt,wintype=2,L=0,D=0,subdrift=False):
    """
    Calculates the power spectral density. The input data is in ind1,
    dt is the time interval between two data points. The parameter
    wintype determines, which window is applied. The parameter L is the length
    of a segment. If L=0 there is only one segment with L=len(ind1). D is the
    overlap between segments. If D=0 it will default to D=L//2
    """
    N = len(ind1)
    if L==0: L=N
    if D==0: D=L//2
    ind1=ind1-np.mean(ind1)
    window = calc_win(L,wintype)
    wss = norm_win(window)
    startix=get_segment_starts(N,L,D)
    cum=[]
    cumc=0;
    for six in startix:
        eix=six+L
        if subdrift:
            le = len(ind1[six:eix])
            xx = np.linspace(-1,1,le)
            pf = np.polyfit(xx,ind1[six:eix],2)
            res = ind1[six:eix]-np.poly1d(pf)(xx)
            newind1 = np.multiply(res,window)
        else:
            newind1 = np.multiply(ind1[six:eix],window)
        myfft   = np.fft.rfft(newind1)
        myfft   = np.multiply(myfft,dt*math.sqrt(1.0/wss))
        Sxx     = np.power(np.abs(myfft),2)
        Sxx     = Sxx/dt/L
        Gxx     = Sxx
        Gxx[1:-1]=Gxx[1:-1]*2  # times 2, since we calculate the one sided psd
        if len(cum)==0:
            cum=Gxx
        else:
            cum=[cum[n]+Gxx[n] for n in range(len(Gxx))]
        cumc+=1
    if cumc>1:
        cum = [cum[n]/cumc for n in range(len(cum))]
    myfreq=np.array([i*1.0/2.0/dt/(len(cum)-1) for i in range(len(cum))])
    newtup=(cum,myfreq)
    return newtup


def mypsa(ind1,dt,wintype=2,L=0,D=0,subdrift=False):
    """
    calculates the power spectral amplitude.
    The input data is in ind1,
    dt is the time interval between two data points. The parameter
    wintype determines, which window is applied. The parameter L is the length
    of a segment. If L=0 there is only one segment with L=len(ind1). D is the
    overlap between segments. If D=0 it will default to D=L//2
    Returns psa,f
    """
    psd,f=mypsd(ind1,dt,wintype,L,D,subdrift)
    psa=np.sqrt(psd)
    newtup=(psa,f)
    return newtup
