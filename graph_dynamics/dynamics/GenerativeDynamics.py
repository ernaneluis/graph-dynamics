'''
Created on Jun 9, 2017

@author: cesar
'''




import math
import time
import pylab
import random
import numpy as np
import networkx as nx
from scipy import stats
from scipy.stats import pareto, norm, bernoulli

from graph_dynamics.dynamics.datatypes import GraphsDynamics

class ErnaneBitcoinDynamics(GraphsDynamics):
    
    def __init__(self,initial_network):
        name_string = "GammaProcess"
        type_of_dynamics="SnapShot"
        GraphsDynamics.__init__(self,initial_network,type_of_dynamics)
        
        self.GRAPH = initial_network
        self.make_nodes_active()





    ######################### end config variables #########################  
    def generate_network_series(self,number_of_steps,output_type):
        """
        Applies the kernel function on every pair of data points between :param x and :param x1.

        In case when :param x1 is None the kernel is applied on every pair of points in :param x.
        :param x: first set of points
        :param x1: second set of points
        :return: distance between every two points
        """
        
        graph_series = [self.intital_graph]
        for T in range(1,number_of_steps):
            graph_series.append(self.evolve_function(graph_series[T-1]))
    
       
    def evolve_function(self,dynamical_process=None):
        """
        """
        self.make_nodes_active()
        
    def define_network_series(self,network_paths,output_type,):
        """
        """
        return None
       
    def make_node_active(self, n):
        ## assign a node attribute nonactive or active
        # is sample the activity once and do the bernoully sample at each time step
        activity_firing_rate = self.GRAPH.node[n]['activity_firing_rate']
        if (activity_firing_rate > 1):
            activity_firing_rate = 1 / activity_firing_rate
        # set if a node is active or not
        self.GRAPH.node[n]['type'] = bernoulli.rvs(activity_firing_rate)
        
    def make_nodes_active(self):
        for n in self.NODES:
            self.make_node_active(n)
        return self    
        
        