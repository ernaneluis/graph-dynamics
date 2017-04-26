'''
Created on Apr 17, 201

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
import sys

import matplotlib 

matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['text.usetex'] = True 

class PallaDynamics:
    
    def __init__(self,alpha,tau,phi,rho,lamb,lamb_parameters,lamb_maximum):
        """
        Here we follow
        
        Bayesian Nonparametrics for Sparse Dynamic Networks
        Konstantina Palla, Francois Caron, Yee Whye Teh
        2016
        
        """
        self.alpha = alpha
        self.tau = tau
        self.phi = phi
        self.rho = rho
        
        self.lamb = lamb
        self.lamb_parameters = lamb_parameters
        self.lamb_maximum = lamb_maximum
        self.gamma = quadrature(self.lamb, 0., self.alpha, self.lamb_parameters)[0]
        self.processDefined = False
    
    def generateInitialNetwork(self,numberOfNodes):
        G = process.GammaProcess(self.alpha,self.tau,self.lamb,self.lamb_parameters,self.lamb_maximum)
        G_measure = G.stickBreakingConstruction(numberOfNodes)
        self.old_interactions = np.zeros((numberOfNodes,numberOfNodes))
        self.network = nx.Graph()
        for node_i in range(numberOfNodes):
            for node_j in range(numberOfNodes):
                if node_i != node_j:
                    self.old_interactions[node_i,node_j] = poisson.rvs(2*G_measure[0][node_i]*G_measure[0][node_j])
                    if self.old_interactions[node_i,node_j] > 0:
                        self.network.add_edge(node_i,node_j)
                else:
                    self.old_interactions[node_i,node_j] = poisson.rvs(G_measure[0][node_i]*G_measure[0][node_j])
                    if self.old_interactions[node_i,node_j] > 0:
                        self.network.add_edge(node_i,node_j)
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
        """
        CT = []
        
        C = []
        for k in range(K1):
            C.append(1.)
        W_seen = []
        
        for c in C:
            W_seen.append(gamma_distribution.rvs(c,self.phi*self.tau))
    
    def CRP(self,numberOfCostumers):
        """
        Chinese restaurant process
        """
        theta = np.random.choice(self.inhomogeneousPoisson())
        Thetas = [theta]
        p = [1./(self.alpha + 1), self.alpha/(self.alpha + 1)]
        numberOfSeatedCostumers = [1.]
        numberOfTables = 1
        for i in range(numberOfCostumers-1):
            p = np.concatenate([np.array(numberOfSeatedCostumers),[self.alpha]])/(self.alpha + sum(numberOfSeatedCostumers))
            selectedTable = np.random.choice(np.arange(numberOfTables+1),p=p)
            if selectedTable == numberOfTables:
                #NewTableSelected
                numberOfTables += 1
                theta = np.random.choice(self.inhomogeneousPoisson())
                numberOfSeatedCostumers.append(1.)
                Thetas.append(theta)
                
            else:
                numberOfSeatedCostumers[selectedTable] = numberOfSeatedCostumers[selectedTable] + 1.
        
        return (Thetas,numberOfSeatedCostumers)
    
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