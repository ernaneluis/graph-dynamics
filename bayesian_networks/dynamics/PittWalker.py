'''
Created on Apr 17, 2017

@author: cesar
'''

import numpy as np
from matplotlib import pyplot as plt
from bayesian_networks.random_measures import process
from scipy.stats import poisson, beta, levy_stable, expon
from scipy.special import gamma
from scipy.stats import gamma as gamma_distribution
from scipy.integrate import quadrature
import networkx as nx
import copy as copy
import sys

import matplotlib 

matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['text.usetex'] = True 

class PallaDynamics:
    
    def __init__(self,sigma,tau,alpha,phi,rho,lamb,lamb_parameters,lamb_maximum):
        """
        Here we follow
        
        Bayesian Nonparametrics for Sparse Dynamic Networks
        Konstantina Palla, Francois Caron, Yee Whye Teh
        2016
        
        """
        #MEASURE RELATED VARIABLES
        self.alpha = alpha
        self.tau = tau
        self.sigma = sigma 
        
        #DYNAMICAL VARIABLES
        self.phi = phi
        self.rho = rho
        
        self.lamb = lamb
        self.lamb_parameters = lamb_parameters
        self.lamb_maximum = lamb_maximum
        
        self.gamma = quadrature(self.lamb, 0., self.alpha, self.lamb_parameters)[0]
        
        self.processDefined = False
    
    def generateInitialNetwork(self,numberOfNodes):
        #TO BE SUBSTITUTED ACCORDING TO BFRY PRIORS
        G = process.GammaProcess(self.alpha,self.tau,self.lamb,self.lamb_parameters,self.lamb_maximum)
        W, Theta  = G.stickBreakingConstruction(numberOfNodes)
         
        #HERE WE HAVE THE POISSON RANDOM VARIABLES
        self.old_interactions = np.zeros((numberOfNodes,numberOfNodes))
        
        #HERE WE HAVE THE C PROCESS
        self.C_old1 = [poisson.rvs(self.phi*W[node_i]) for node_i in range(len(W))]
        self.ThetaC_old1 = copy.copy(Theta)
        
        #THE NETWORKX GRAPH FOR PLOTTING AND REFERENCE
        self.network = nx.Graph()
        for node_i in range(numberOfNodes):
            
            for node_j in range(numberOfNodes):
                if node_i != node_j:
                    self.old_interactions[node_i,node_j] = poisson.rvs(2*W[node_i]*W[node_j])
                    if self.old_interactions[node_i,node_j] > 0:
                        self.network.add_edge(node_i,node_j)
                else:
                    self.old_interactions[node_i,node_j] = poisson.rvs(W[node_i]*W[node_j])
                    if self.old_interactions[node_i,node_j] > 0:
                        self.network.add_edge(node_i,node_j)
                        
        #THIS GENERATES THE C PROCESS WHICH CORRESPONDS TO THE CONTINUOS PART OF THE MEASURE (EMPTY CHAIRS FOR CRP)
        w_total_mass = gamma.rvs(self.alpha,self.tau + self.phi)
        number_of_costumers = poisson.rvs(self.phi*w_total_mass)
        self.ThetasC_old2, self.C_old2 = self.CRP(number_of_costumers)
        
        return self.network
        
    def normalizedLamb(self,x):
        """
        normalized version of B0 for the inhomogeneous Poisson Process
        """ 
        return (1./self.gamma)*self.lamb(x,*self.lamb_parameters)
    
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
        test = 0
        arrivals = []
        while len(arrivals) == 0: 
            rateBound = self.lamb_maximum/self.gamma
            T = self.alpha
            J = poisson.rvs(T * rateBound)
            datesInSeconds = np.random.uniform(0., T, J)
            intensities = self.normalizedLamb(datesInSeconds) / rateBound
            r = np.random.uniform(0., 1., J)
            arrivals = np.take(datesInSeconds, np.where(r < intensities)[0])
            test += 1.
            if(test > 10):
                sys.exit(1) 
        return arrivals
    
    def initializeDynamics(self,K1):
        """
        
        #THE NETWORKX GRAPH FOR PLOTTING AND REFERENCE
        self.network = nx.Graph()
        self.old_interactions = np.zeros((numberOfNodes,numberOfNodes))
                        
        #THIS GENERATES THE C PROCESS WHICH CORRESPONDS TO THE CONTINUOS PART OF THE MEASURE (EMPTY CHAIRS FOR CRP)
        w_total_mass = gamma.rvs(self.alpha,self.tau + self.phi)
        number_of_costumers = poisson.rvs(self.phi*w_total_mass)
        self.ThetasC_old2, self.C_old2 = self.CRP(number_of_costumers)
        
        """
        CT = []
        
        C = []
        for k in range(K1):
            C.append(1.)
        W_seen = []
        
        for c in C:
            W_seen.append(gamma_distribution.rvs(c,self.phi*self.tau))
    

    
    def hiddenCPath(self,T,K1):
        """
        """
        CT = []
        #=============================
        # Initialization
        #=============================
        C0_observed = []
        for k in range(K1):
            C0_observed.append(1.)
        C0_unobserved = []
        #=============================
        # TIME LOOP
        #=============================
        CT.append((C0_observed,C0_unobserved))
        for t in range(T):
            C1_observed = []
            for c in CT[t][0]:
                w_seen = gamma_distribution.rvs(c,self.phi+self.tau)
                C1_observed.append(poisson(self.phi*self.tau))
            
            w_unobserved = gamma_distribution.rvs(self.alpha,self.phi+self.tau)
            c_unobserved_mass = poisson.rvs(self.phi*w_unobserved) 