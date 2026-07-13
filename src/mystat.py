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
    
    
def meanerr(meanvals, errvals): 
    xsum = 0
    errsum = 0
    for x,err in zip(meanvals,errvals):
        xsum = xsum + x / err /err
        errsum = errsum + 1.0/err/err
    mean = xsum /errsum
    err = math.sqrt(1.0/errsum)
    return mean,err
    
def weightedMean(vals,errs):
    """
    vm,vm_err,chi2 = weightedMean(vals,errs)
    Calculates the weighted mean from an input 1d array vals
    and a second 1d input array erros
    Output: 
        vm = mean value, 
        vm_err = error value
        chi2 = chi2
        Note if chi2> NDF, one can scale the errors by sqrt(chi2/NDF)
    """

    vm  = np.sum(vals/(errs*errs))/np.sum(1.0/(errs*errs))
    vm_err = np.sqrt(1.0/np.sum(1.0/(errs*errs)))
    chi2 = np.sum((vals-vm)**2/errs**2)
    return vm,vm_err,chi2


def meanWithCov(vals,cov):
    """
    mm,err,chi2 = meanWithCov(vals,cov)
    Calculates the mean from an input 1d array andthe input's covariance
    matrix cov: vals = (1xN) np.array, cov = (NxN) np.array
    Output: 
        mm = mean value taking into account the covariances
        err = error of the mean
        chi2 = chi2
        Note if chi2> NDF, one can scale the errors by sqrt(chi2/NDF)
    
    """

    A = np.ones((len(vals),1))
    cov_inverse = np.linalg.inv(cov)
    T= np.dot(A.T,cov_inverse)/np.dot(A.T,np.dot(cov_inverse,A))
    M = np.dot(T,vals)
    cov_out  = np.dot(np.dot(T,cov),T.T)
    chi2 = np.dot(np.dot((vals-M).T,cov_inverse),vals-M)
    mm=M[0]
    err=cov_out[0,0]
    return mm,err,chi2
    