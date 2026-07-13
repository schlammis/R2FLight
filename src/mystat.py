#!/usr/bin/env python
""" A module to the AllanDeviation and other
statistical measures


"""


__author__          =   "Stephan Schlamminger"
__email__           =   "schlammi@gmail.com"
__status__          =   "Development"
__date__            =   "08/28/11"
__version__         =   "0.1"

import numpy as np
import math


def AllanVariance(d,s=None):
    """
    s, allanvariance, erro_on_allanvariance = AllanVariance(d,s=None)
    """
    x=1
    if s is None:
        s=[]
        while x<=len(d)/4:
            s.append(x)
            x=x*2
    N=len(d)
    allan=[]    
    allanerr=[]
    for tau in s:
        ybar=[]
        co=0
        while co+tau<N:
            newybar=np.average(d[co:co+tau])
            ybar.append(newybar)
            co=co+tau
        co=0
        avar=[]
        while co+1<len(ybar):
            avar.append( (ybar[co+1]-ybar[co])*(ybar[co+1]-ybar[co])/2 )           
            co=co+1
        allan.append(np.mean(avar))
        allanerr.append(np.std(avar,ddof=1)/math.sqrt(len(avar)))
    #allan=np.array(allan)/np.mean(d)
    return np.array(s),np.array(allan),np.array(allanerr)
    
    
def AllanDeviation(d,s=None):
    """
    s, allandeviation, erro_on_allandev = AllanDeviation(d,s=None)
    """

    s, va, vaerr = AllanVariance(d, s)
    std = np.sqrt(va)
    stderr = 0.5 / np.sqrt(va) * vaerr
    return s, std, stderr
