'''
Created on Sep 20, 2017

@author: cesar
'''

from scipy.stats import norm, pareto, lognorm, gamma, weibull_min, weibull_max, gengamma, expon
import numpy as np
from scipy.stats import kstest
import sys
from matplotlib import pyplot as  plt
from scipy import interpolate
from scipy.integrate import quadrature
from scipy.stats import expon, lognorm, pareto


import numpy as np
import pandas as pd

import statsmodels.api as sm
from arch.univariate import ARCH, GARCH, ConstantMean
from arch.univariate import ARX
from arch import arch_model

from matplotlib import pyplot as plt

def log_difference(time_series):
    return ((np.roll(time_series,-1) - time_series)[:-1])/time_series[:-1]


def volatilities(time_series,n=30,absolute=True):
    """
    Here we follow:
    
    The Statistical Properties of the Volatility of Price Fluctuations
    Liu Y. et al
    
    return volatilities (pandas series)
    """
    N = len(time_series)
    volatility = []
    volatility_index = []
    if absolute:
        time_series = abs(time_series)
    for t in range(N-n):
        bag = []
        for i in range(n):
            bag.append(time_series.values[t+i])
        volatility.append(np.asarray(bag).mean())
        volatility_index.append(time_series.index[t])
    volatility = pd.Series(volatility,index=volatility_index,name="volatility")
    return volatility


#STATISTICS FITS


def bestFitScaled(X,plot=False,distToTest=["norm","pareto","lognorm","gamma",
                                     "expon","weibull_min","weibull_max"],
            numberOfBins=100,logScaled=False,theLabel="something",color="red",symbol="o"):
    """
    Version of August 2015
    
    X: data
    
    return 
    """
    xMin = min(X)
    xMax = max(X)
    dX = (xMax - xMin)*0.01
    
    support = np.arange(xMin,xMax,dX)
    bestFit = []
    error = []
            
    if("pareto" in distToTest):
        try:
            dist_param = pareto.fit(X)
            myDist = pareto(*dist_param)
            kt, p_value =  kstest(X,"pareto",dist_param)
            bestFit.append({"distribution":"pareto","ktest":kt,"pvalue":p_value,"parameters":dist_param})
            if(plot):
                Y = myDist.pdf(support)
                plt.plot(support,Y,linewidth=2.0)
                if logScaled:
                    hist, bins = np.histogram(X,bins=numberOfBins,normed=True)
                    plt.plot(bins[:-1],hist,symbol,color=color,label=theLabel)
                #stuff = plt.hist(X,bins=numberOfBins,normed=True)
        except:
            error.append(("pareto_err",sys.exc_info()))
            
    if("lognorm" in distToTest):
        try:
            dist_param = lognorm.fit(X)
            myDist = lognorm(*dist_param)
            kt, p_value =  kstest(X,"lognorm",dist_param)
            bestFit.append({"distribution":"lognorm","ktest":kt,"pvalue":p_value,"parameters":dist_param})
            if(plot):
                Y = myDist.pdf(support)
                plt.plot(support,Y,color=color,linewidth=2.0)
                if logScaled:
                    hist, bins = np.histogram(X,bins=numberOfBins,normed=True)
                    plt.plot(bins[:-1],hist,symbol,color=color,label=theLabel)
                    
        except:
            error.append(("lognorm_err",sys.exc_info()))
    
    #FINISH PLOT
    if(plot):
        if(logScaled==True):
            plt.yscale("log")
            plt.xscale("log")
            
        plt.legend(loc="best")
        plt.show()
    
        
    return (bestFit,error)

def bestFit(X,plot=False,distToTest=["norm","pareto","lognorm","gamma","expon","weibull_min","weibull_max"],numberOfBins=100):
    """
    
    Version of August 2015
    X: data
    
    return 
    """
    xMin = min(X)
    xMax = max(X)
    dX = (xMax - xMin)*0.01
    
    support = np.arange(xMin,xMax,dX)
    bestFit = []
    error = []
    if("norm" in distToTest):
        try:
            dist_param = norm.fit(X)
            myDist = norm(*dist_param)
            kt, p_value =  kstest(X,"norm",dist_param)
            bestFit.append({"distribution":"norm","ktest":kt,"pvalue":p_value,"parameters":dist_param})
            if(plot):
                Y = myDist.pdf(support)
                plt.plot(support,Y,label="norm",linewidth=2.0)
                stuff = plt.hist(X,bins=numberOfBins,normed=True)
        except:
            error.append(("norm_err",sys.exc_info()))
            
    if("pareto" in distToTest):
        try:
            dist_param = pareto.fit(X)
            myDist = pareto(*dist_param)
            kt, p_value =  kstest(X,"pareto",dist_param)
            bestFit.append({"distribution":"pareto","ktest":kt,"pvalue":p_value,"parameters":dist_param})
            if(plot):
                Y = myDist.pdf(support)
                plt.plot(support,Y,label="pareto",linewidth=2.0)
                stuff = plt.hist(X,bins=numberOfBins,normed=True)
        except:
            error.append(("pareto_err",sys.exc_info()))
            
    if("lognorm" in distToTest):
        try:
            dist_param = lognorm.fit(X)
            myDist = lognorm(*dist_param)
            kt, p_value =  kstest(X,"lognorm",dist_param)
            bestFit.append({"distribution":"lognorm","ktest":kt,"pvalue":p_value,"parameters":dist_param})
            if(plot):
                Y = myDist.pdf(support)
                plt.plot(support,Y,label="lognorm",linewidth=2.0)
                stuff = plt.hist(X,bins=numberOfBins,normed=True)
        except:
            error.append(("lognorm_err",sys.exc_info()))
    
    if("gamma" in distToTest):
        try:
            dist_param = gamma.fit(X)
            myDist = gamma(*dist_param)
            kt, p_value =  kstest(X,"gamma",dist_param)
            bestFit.append({"distribution":"gamma","ktest":kt,"pvalue":p_value,"parameters":dist_param})
            if(plot):
                Y = myDist.pdf(support)
                plt.plot(support,Y,label="gamma",linewidth=2.0)
                stuff = plt.hist(X,bins=numberOfBins,normed=True)
        except:
            error.append(("gamma_err",sys.exc_info()))
    
    if("weibull_min" in distToTest):
        try:
            dist_param = weibull_min.fit(X)
            myDist = weibull_min(*dist_param)
            kt, p_value =  kstest(X,"weibull_min",dist_param)
            bestFit.append({"distribution":"weibull_min","ktest":kt,"pvalue":p_value,"parameters":dist_param})
            if(plot):
                Y = myDist.pdf(support)
                plt.plot(support,Y,label="weibull_min",linewidth=2.0)
                stuff = plt.hist(X,bins=numberOfBins,normed=True)
        except:
            error.append(("weibull_min_err",sys.exc_info()))
            
    if("weibull_max" in distToTest):
        try:
            dist_param = weibull_max.fit(X)
            myDist = weibull_max(*dist_param)
            kt, p_value =  kstest(X,"weibull_max",dist_param)
            bestFit.append({"distribution":"weibull_max","ktest":kt,"pvalue":p_value,"parameters":dist_param})
            if(plot):
                Y = myDist.pdf(support)
                plt.plot(support,Y,label="weibull_max",linewidth=2.0)
                stuff = plt.hist(X,bins=numberOfBins,normed=True)
        except:
            error.append(("weibull_max_err",sys.exc_info()))
            
    if("expon" in distToTest):
        try:
            dist_param = expon.fit(X)
            myDist = expon(*dist_param)
            kt, p_value =  kstest(X,"expon",dist_param)
            bestFit.append({"distribution":"expon","ktest":kt,"pvalue":p_value,"parameters":dist_param})
            if(plot):
                Y = myDist.pdf(support)
                plt.plot(support,Y,label="expon",linewidth=2.0)
                stuff = plt.hist(X,bins=numberOfBins,normed=True)
        except:
            error.append(("expon_err",sys.exc_info()))
    #FINISH PLOT
    if(plot):
        plt.legend(loc="best")
        plt.show()
    return (bestFit,error)

def realBestFit(X,plot=False,distToTest=["norm","pareto","expon","lognorm","gamma","weibull_min","weibull_max"],numberOfBins=100):
    """
    """
    bF = bestFit(X,False,distToTest,numberOfBins)[0]
    a = [(b["ktest"],b["distribution"]) for b in bF]
    a.sort()
    return a[0][1]