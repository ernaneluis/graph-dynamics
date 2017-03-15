'''
Created on Mar 13, 2017

@author: cesar
'''
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import poisson, beta, gamma, levy_stable
from scipy.special import gamma
from scipy.integrate import quadrature

from bayesian_networks.utils import numerics

class BetaProcess:
    
    def __init__(self,c,Omega,B0,B0parameters,B0maximum):
        """
        c: concentration parameter
        Omega: we assume that the space were the measure B0 is defined is given by [0,Omega] \subsect of R
        B0: base measure for the beta process
        
        """
        self.B0 = B0
        self.B0parameters = B0parameters
        self.B0maximum = B0maximum
        self.Omega = Omega
        self.c = c
        self.gamma = quadrature(self.B0, 0., Omega, self.B0parameters)[0]
        self.processDefined = False
        
    def normalizedB0(self,x):
        """
        normalized version of B0 for the inhomogeneous Poisson Process
        """ 
        return (1./self.gamma)*self.B0(x,*self.B0parameters)
    
    
    def inhomogeneousPoisson(self):
        """
        generates a set of arrivals from a functional form
        using the thinning process
    
        Parameters:
        T: float
        dT: float
        function: function
        functionParameters
        """
        rateBound = self.B0maximum/self.gamma
        T = self.Omega
        J = poisson.rvs(T * rateBound)
        datesInSeconds = np.random.uniform(0., T, J)
        intensities = self.normalizedB0(datesInSeconds) / rateBound
        r = np.random.uniform(0., 1., J)
        arrivals = np.take(datesInSeconds, np.where(r < intensities)[0])
        return arrivals
    
    def generateBetaProcess(self,N):
        """
        Algorithm from 
        
        Hierarchical Beta Process and the Indian Buffet Process
        Romain Thibaux
        Michael I. Jordan
        """
        P = []
        W = []
        for i in range(1,N):
            lamb = (self.gamma*self.c)/(self.c+i-1.)
            K1 = poisson.rvs(lamb)
            newLocations = []
            while len(newLocations) < K1:
                newLocations.extend(self.inhomogeneousPoisson())
            W.append(newLocations[:K1])   
            P.append(beta.rvs(1,self.c + i - 1.,size=K1))
        self.P = np.concatenate(P)
        self.W = np.concatenate(W)
        self.processDefined = True
        return (P,W)
                
    def plotProcess(self,plotName=None,saveTo=None): 
        """
        """
        ymin = np.zeros(len(self.W))
        plt.vlines(self.W, ymin, self.P)
        plt.plot(self.W,self.P,"ro",markersize=12)
        plt.grid(True)
        plt.show()
        

class GeneralizedGammaProcess:
    """
    Here we use the urn representation as defined in 
    
    Sparse Graph USING Exchangable Random Measures
    Francois Caron 
    Emily B. Fox
    """
    
    def __init__(self,H0,sigma):
        self.H0
        self.sigma = sigma
    
    def EPPF(self,m,n,t):
        """
        Exchangable Partitiopn Probablity Function
        
        m -- vector of counts
        """
        k = len(m)
        
        def eppf_integrand(s):
            return np.power(s,n-k*self.sigma-1)*levy_stable.pdf(t-s)
        
        g = levy_stable.pdf(t)
        A = (self.sigma**k)*(t**(-n))
        B = gamma(n-k*self.sigma)
        C = quadrature(eppf_integrand, 0, t)[0]
        
        a = np.prod(gamma(m - self.sigma))
        b = gamma(1 - self.gamma)**k
        
        X = a/b
        