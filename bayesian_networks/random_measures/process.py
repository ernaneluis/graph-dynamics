'''
Created on Mar 13, 2017

@author: cesar
'''
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import poisson, beta, levy_stable, expon
from scipy.special import gamma
from scipy.stats import gamma as gamma_distribution
from scipy.integrate import quadrature
import matplotlib 

matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['text.usetex'] = True 


#======================================
#  PROCESS 
#======================================

class GammaProcess:
    
    def __init__(self,alpha,tau,lamb,lamb_parameters,lamb_maximum):
        """
        """
        self.alpha = alpha
        self.tau = tau
        self.lamb = lamb
        self.lamb_parameters = lamb_parameters
        self.lamb_maximum = lamb_maximum
        self.gamma = quadrature(self.lamb, 0., self.alpha, self.lamb_parameters)[0]
        self.processDefined = False
    
    def normalizedLamb(self,x):
        """
        normalized version of B0 for the inhomogeneous Poisson Process
        """ 
        return (1./self.gamma)*self.lamb(x,*self.lamb_parameters)
    
    def stickBreakingConstruction(self,K):
        """
        Here we follow the algorithm of 
        
        Gamma Processes, Stick Breaking and Variational Inference
        Anirban Roychowdhury
        Brian Kulis
        
        K: is the truncation parameter and indicates the number of atoms accepted for the algorithm
        """
        W = []
        P = []
        while len(W) < K:
            W.extend(self.inhomogeneousPoisson())   
        W = W[:K]
        
        k = 0 
        roundNumber = 1
        while k < K:
            K1 = poisson.rvs(self.gamma)
            for i in range(K1):
                Ek = expon.rvs(self.tau)
                Tk = gamma_distribution.rvs(roundNumber,self.alpha)
                P.append( Ek*np.exp(-Tk) ) 
                k+=1
                if k == K:
                    break
            roundNumber += 1
        self.processDefined = True
        
        self.P = P
        self.W = W
        return (P,W)
                            
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
        rateBound = self.lamb_maximum/self.gamma
        T = self.alpha
        J = poisson.rvs(T * rateBound)
        datesInSeconds = np.random.uniform(0., T, J)
        intensities = self.normalizedLamb(datesInSeconds) / rateBound
        r = np.random.uniform(0., 1., J)
        arrivals = np.take(datesInSeconds, np.where(r < intensities)[0])
        return arrivals
    
    def plotProcess(self,plotName=None,saveTo=None): 
        """
        """
        ymin = np.zeros(len(self.W))
        plt.vlines(self.W, ymin, self.P)
        plt.plot(self.W,self.P,"ro",markersize=12)
        plt.grid(True)
        plt.show()
        
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
     