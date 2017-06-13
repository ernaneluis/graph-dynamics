'''
Created on Jun 9, 2017

@author: cesar
'''

import networkx as nx
import random
import pylab
import numpy as np
from scipy import stats
from scipy.stats import pareto, norm, bernoulli
import math
import time

class GenerateErnaneGraphs():
        
        def __init__(self,numberOfNodes,numberOfConnections,activity_gamma,RESCALING_FACTOR,x_min):
            ######################### config variables #########################
            
            self.N = numberOfNodes
            self.gamma = activity_gamma
            ## calculating the activity potential following pareto distribution
            self.x_min              = x_min  # thershold
            self.X                  = pareto.rvs(self.gamma, loc=self.x_min, size=self.N)  # get N samples from  pareto distribution
            self.X                  = self. X / max(self.X)  # every one smaller than one
            self.activity_potential = np.take(self.X, np.where(self.X > self.x_min)[0])  # using the thershold
            ######################### config variables #########################
            self.M                  = numberOfConnections  # number of edges a node can connect
            
            #self.avrgActiveNodes    = self.N * 0.05  # 5% from total N nodes
            # average of Total number of Actives Nodes per time, this is empirical n<x>N
            self.RESCALING_FACTOR   = RESCALING_FACTOR # or avrgActiveNodes/(activity_potential.mean()*N)
    
            ## calculating the activity potential following pareto distribution
            self.X                  = pareto.rvs(self.gamma, loc=self.x_min, size=self.N)  # get N samples from  pareto distribution
            self.X                  = self. X / max(self.X)  # every one smaller than one
            self.activity_potential = np.take(self.X, np.where(self.X > self.x_min)[0])  # using the thershold
            
    
            # calculate the wealth distribution following pareto law
            # creating graph
            self.GRAPH              = nx.Graph()
            self.NODES              = list(xrange(self.N))  # creating list of nodes with index from 0 de N-1
    
            ######################### end config variables #########################
            
            self.GRAPH.add_nodes_from(self.NODES)  # adding to the graph

            # run over all nodes to set initial attributes
            for n in self.NODES:
                ## what is the purpose of rescaling factor?
                # ai = xi*n => probability per unit time to create new interactions with other nodes
                # activity_firing_rate is an probability number than [0,1]
                self.GRAPH.node[n]['activity_firing_rate'] = self.activity_potential[n] * self. RESCALING_FACTOR
    
                # With probability ai*delta_t each vertex i becomes active and generates m links that are connected to m other randomly selected vertices
                self.GRAPH.node[n]['activity_probability'] = self.GRAPH.node[n]['activity_firing_rate'] * self.DELTA_T
    
                # no walkers
                self.GRAPH.node[n]['walker'] = 0
    
                # setting the node the initial amount of wealth
                self.GRAPH.node[n]['amount'] = self.amount[n]
        
        #return self.GRAPH
    
    
