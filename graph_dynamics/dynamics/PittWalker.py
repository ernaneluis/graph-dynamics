'''
Created on Apr 17, 2017

@author: cesar
'''

import numpy as np
from matplotlib import pyplot as plt
from graph_dynamics.random_measures import process
from graph_dynamics.random_measures.normalized_process import ChineseRestaurantProcess
from scipy.stats import poisson, beta, levy_stable, expon, binom
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
        
    def __generateHiddenPath(self,T):
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
        Here this is a network path without memory, we only make use of the changes in the W measures
        """
        C_TimeSeries = self.__generateHiddenPath(T)
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
    
    def __forgetInteractions(self,currentCaronNetwork):
        """
        Here we assume that the time increment \Delta_t = 1
        """
        for node, neigh in currentCaronNetwork.edge.iteritems():
            for ng, weight in neigh.iteritems():
                currentCaronNetwork[node][ng]["weight"] = binom.rvs(weight["weight"],np.exp(-self.rho))

    
    def __updateInteractions(self,currentCaronGraph,sigma_increment=0.,tau_increment=0.,table_and_costumers=None):
        """
        Here we generate from the normalized measure trough a chinese restaurant process
        """
        currentGraph = currentCaronGraph.network
        self.full_graph_measure = gamma.rvs(self.sigma+sigma_increment,self.tau+tau_increment) 
        self.number_of_arrivals =  poisson.rvs(self.full_graph_measure**2.) # THIS CORRESPONDS TO d_t^*
        
        costumer_seats,Thetas,numberOfSeatedCostumers = currentCaronGraph.measure.normalized_random_measure(self.number_of_arrivals*2,
                                                                                               table_and_costumers)
        for k in range(self.number_of_arrivals):
            Uk1 = costumer_seats[2*k]
            Uk2 = costumer_seats[2*k+1]
            try:
                w = currentGraph.edge[Uk1][Uk2]["weight"]
                currentGraph.add_edge(Uk1,Uk2,weight=w+1)
            except:
                currentGraph.add_edge(Uk1,Uk2,weight=1)

        return currentGraph
    
    
    def generateNetworkPaths_1(self,T):
        """
        This is the network path from the original paper
        
        T: int
            number of discrete time steps in the evolution
        """
        C_TimeSeries = self.__generateHiddenPath(T)
        Networks_TimeSeries = []
        Networks_TimeSeries.append(self.CaronFoxGraph_0.network)
        for t in range(1,T):
            print "Time Iteration {0}".format(t)
            C_t_1 = C_TimeSeries[t-1] 
            C_t = C_TimeSeries[t]
            
            # we organize the atoms of the hidden process
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
            self.__forgetInteractions(self.CaronFoxGraph_0.network)
            self.__updateInteractions(self.CaronFoxGraph_0,
                                      sigma_increment,
                                      tau_increment,
                                      tables_and_costumers)
            
            
            print "Number of current nodes {0}".format(self.CaronFoxGraph_0.network.number_of_nodes())
            Networks_TimeSeries.append(self.CaronFoxGraph_0.network)
            
        return Networks_TimeSeries
    
    
    