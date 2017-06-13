'''
Created on Apr 17, 2017

@author: cesar
'''

import numpy as np
from matplotlib import pyplot as plt
from graph_dynamics.random_measures import process
from graph_dynamics.random_measures.normalized_process import ChineseRestaurantProcess
from scipy.stats import poisson, beta, levy_stable, expon
from scipy.special import gamma as gamma_function
from scipy.stats import gamma
from scipy.integrate import quadrature
import networkx as nx
import copy as copy
import sys

import matplotlib 

matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['text.usetex'] = True 

class PallaDynamics:
    
    def __init__(self,phi,rho,CaronFoxGraph):
        """
        Here we follow:
        
        Bayesian Nonparametrics for Sparse Dynamic Networks
        Konstantina Palla, Francois Caron, Yee Whye Teh
        2016
        
        """
        #DYNAMICAL VARIABLES
        self.phi = phi
        self.rho = rho
        
        #GRAPH
        self.CaronFoxGraph_0 = CaronFoxGraph
        self.sigma = self.CaronFoxGraph_0.sigma
        self.tau = self.CaronFoxGraph_0.tau
        self.alpha = self.CaronFoxGraph_0.alpha 
        
    def generateHiddenPath(self,T):
        """
        """
        C_TimeSeries = []
        
        #FROM W_ATOMS        
        C_measure = {}
        for i,w in enumerate(self.CaronFoxGraph_0.measure.W):
            new_c = poisson.rvs(self.phi*w)
            if new_c != 0:
                C_measure[self.CaronFoxGraph_0.measure.W[i]] = new_c
        
        # NEW ATOMS
        w_star = gamma.rvs(self.alpha,self.tau + self.phi)
        number_of_costumers = poisson.rvs(w_star*self.phi)
        costumer_seats,Thetas_2,C_2 = ChineseRestaurantProcess(number_of_costumers,
                                                               self.CaronFoxGraph_0.measure.lambda_measure)
        for theta, c in zip(Thetas_2,C_2):
            C_measure[theta] = c
        
        print C_measure
        C_TimeSeries.append(C_measure)
        for i in range(1,T):
            # FROM OLD ATOMS
            C_measure = {}
            for theta,c in C_TimeSeries[i-1].iteritems():
                new_c = poisson.rvs(gamma.rvs(c,self.tau + self.phi)*self.phi)
                if new_c != 0:
                    C_measure[theta] = new_c
            # NEW ATOMS
            w_star = gamma.rvs(self.alpha,self.tau + self.phi)
            number_of_costumers = poisson.rvs(w_star*self.phi)
            costumer_seats,Thetas_2,C_2 = ChineseRestaurantProcess(number_of_costumers,
                                                                           self.CaronFoxGraph_0.measure.lambda_measure)
            for theta, c in zip(Thetas_2,C_2):
                C_measure[theta] = c
                    
            C_TimeSeries.append(C_measure)
        
        return C_TimeSeries
    
    def generateNetworkPaths(self,T):
        """
        """
        C_TimeSeries = self.generateHiddenPath(T)
        Networks_TimeSeries = []
        Networks_TimeSeries.append(self.CaronFoxGraph_0.network)
        for t in range(1,T):
            print "Time Iteration {0}".format(t)
            C_t_1 = C_TimeSeries[t-1] 
            C_t = C_TimeSeries[t]
            
            full_table_set = list(set(C_t.values()).union(set(C_t_1.values())))
            tables_and_costumers = dict(zip(full_table_set,np.zeros(len(full_table_set))))
            for table in tables_and_costumers.keys():
                if table in C_t:
                    tables_and_costumers[table] += C_t[table]
                if table in C_t_1:
                    tables_and_costumers[table] += C_t_1[table]
                    
            sigma_increment = sum(tables_and_costumers.keys())
            tau_increment = 2*self.phi
            print "Number of old tables {0}".format(sigma_increment)
            network = self.CaronFoxGraph_0.generateNetwork(sigma_increment,
                                                                 tau_increment,
                                                                 tables_and_costumers)
            
            print "Number of current nodes {0}".format(network.number_of_nodes())
            Networks_TimeSeries.append(network)
        return Networks_TimeSeries