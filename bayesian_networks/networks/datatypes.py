'''
Created on May 3, 2017

@author: cesar
'''
import copy
import matplotlib
import numpy as np
from networkx import nx
from collections import namedtuple
from matplotlib import pyplot as plt
from scipy.integrate import quadrature
from scipy.stats import poisson, gamma
from abc import ABCMeta, abstractmethod

matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True

#==============================================================
#                           ABSTRACT CLASS 
#==============================================================

class BayesianNetwork(object):
    """
    This class is a superclass for all types of kernels (positive definite functions).
    """
    __metaclass__ = ABCMeta

    def __init__(self,name_string,identifier_string):
        self.name_string = name_string
        self.identifier_string = identifier_string
        
    @abstractmethod
    def inferMeasures(self):
        raise NotImplemented()
    
    @abstractmethod
    def generateNetwork(self):
        raise NotImplemented()
    
#==============================================================
#                           FINITE PROCESS 
#==============================================================
    
class FiniteProcessGraphs(BayesianNetwork):
    """
    This class is a superclass for all types of kernels (positive definite functions).
    """
    __metaclass__ = ABCMeta
    types_of_network = {1:"Undirected",2:"Directed",3:"Bipartite"}
    
    def __init__(self, identifier_string,randomMeasure,network=None,type_of_network=1):
        self.name_string = "FiniteProcessGraph"
        self.identifier_string = identifier_string
        self.measure = randomMeasure
        self.type_of_network = type_of_network
        
        BayesianNetwork.__init__(self,self.name_string,self.identifier_string)
        if network != None:
            print "Network Given"
            self.inferMeasure()
        else:
            self.network = self.generateNetwork()
        print "#============================"
        print "#  Bayesian Network Ready    "
        print "# Number of nodes {0}        ".format(self.network.number_of_nodes())
        print "# Number of edges {0}        ".format(self.network.number_of_edges())
        print "#============================"
        
    def generateNetwork(self):
        """
        We generate the network according to the paper:
        
        Sparse Graphs Using Exchangable Random Measures
        Francois Caron, Emily B. Fox 
        
        But Instead of Using the Normalized Completly Random Measure Approach, we directly 
        use the Weights as defined by the Finite Measure  
        """        
        #THE NETWORKX GRAPH FOR PLOTTING AND REFERENCE
        self.network = nx.Graph()
        for node_i in range(int(self.measure.K)):
            for node_j in range(int(self.measure.K)):
                if node_i != node_j:
                    n_ij = poisson.rvs(2*self.measure.W[node_i]*self.measure.W[node_j])
                    if n_ij > 0:
                        if self.type_of_network == 1:
                            self.network.add_edge(node_i,node_j)
                        if self.type_of_network == 2:
                            self.network.add_edge(node_i,node_j,weight=n_ij)
                else:
                    n_ii = poisson.rvs(self.measure.W[node_i]*self.measure.W[node_j])
                    if n_ii > 0:
                        if self.type_of_network == 1:
                            self.network.add_edge(node_i,node_j)
                        if self.type_of_network == 2:
                            self.network.add_edge(node_i,node_j,weight=n_ii)
                        
        return self.network
    
    #===============================
    # INFERENCE
    #===============================
    def inferMeasures(self):
        return None

#==============================================================
#                           CARON FOX 
#==============================================================

class CaronFoxGraphs(BayesianNetwork):
    """
    This class is a superclass for all types of kernels (positive definite functions).
    """
    __metaclass__ = ABCMeta
    types_of_network = {1:"Undirected",2:"Directed",3:"Bipartite"}
    
    def __init__(self, identifier_string,randomMeasure,network=None,type_of_network=1):
        self.name_string = "CaronFox"
        self.identifier_string = identifier_string
        self.type_of_network = type_of_network 
        self.measure = randomMeasure 
        BayesianNetwork.__init__(self,self.name_string,self.identifier_string)
        if network != None:
            print "Network Given"
            self.inferMeasure()
        else:
            self.network = self.generateNetwork()
            
    def generateNetwork(self,numberOfNodes):
        """
        We generate the network according to the paper:
        
        Sparse Graphs Using Exchangable Random Measures
        Francois Caron, Emily B. Fox 
        
        """        
        #THE NETWORKX GRAPH FOR PLOTTING AND REFERENCE
        self.network = nx.Graph()
        self.old_interactions = np.zeros((numberOfNodes,numberOfNodes))
        

                        
        #THIS GENERATES THE C PROCESS WHICH CORRESPONDS TO THE CONTINUOS PART OF THE MEASURE (EMPTY CHAIRS FOR CRP)
        w_total_mass = gamma.rvs(self.alpha,self.tau + self.phi)
        number_of_costumers = poisson.rvs(self.phi*w_total_mass)
        self.ThetasC_old2, self.C_old2 = self.CRP(number_of_costumers)
        
        return self.network
    
    #===============================
    # INFERENCE
    #===============================
    def inferMeasure(self):
        return None
    
    