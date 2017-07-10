'''
Created on May 3, 2017

@author: cesar
'''


import copy
import cPickle
import matplotlib
import numpy as np
from networkx import nx
from collections import namedtuple
from matplotlib import pyplot as plt
from scipy.integrate import quadrature
from scipy.stats import poisson, gamma
from abc import ABCMeta, abstractmethod
from _pyio import __metaclass__

matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True

class Graph(object):
    """
    """    
    __metaclass__ = ABCMeta

    def __init__(self,name_string,identifier_string):
        self.name_string = name_string
        self.identifier_string = identifier_string
        
    @abstractmethod        
    def get_networkx(self):
        raise NotImplemented()
        
    @abstractmethod      
    def get_adjancency_matrix(self):
        raise NotImplemented()

    @abstractmethod        
    def get_edge_list(self):
        raise NotImplemented()
 
    @abstractmethod       
    def get_number_of_edges(self):
        raise NotImplemented()

    @abstractmethod        
    def get_number_of_nodes(self):
        raise NotImplemented()    
    

class CommunityGraphs(object):
    """
    """  
    __metaclass__ = ABCMeta

    def __init__(self,name_string,identifier_string):
        self.name_string = name_string
        self.identifier_string = identifier_string
        
    @abstractmethod        
    def get_nodes(self):
        raise NotImplemented()
    
#==============================================================
#                           ABSTRACT CLASS 
#==============================================================

class BayesianGraph(object):
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
#                           ABSTRACT CLASS 
#==============================================================

class OwnershipGraph(object):
    """
    This class is a superclass for bipartite graph define as
    costumer-product relationships, initially we except to 
    use these class for recommender systems
    """
    __metaclass__ = ABCMeta

    def __init__(self,name_string,identifier_string):
        self.name_string = name_string
        self.identifier_string = identifier_string
        
    @abstractmethod
    def inferMeasures(self):
        raise NotImplemented()
    
    @abstractmethod
    def generateIndicatorMatrix(self):
        raise NotImplemented()

#==============================================================
#                           ABSTRACT CLASS 
#==============================================================

class FromFileGraph(object):
    """
    This class is a superclass for all types of kernels (positive definite functions).
    """
    __metaclass__ = ABCMeta

    def __init__(self,name_string,identifier_string,graph_files_folder,graph_file_string):
        self.name_string = name_string
        self.identifier_string = identifier_string
        

class CryptocurrencyGraphs(FromFileGraph):
    __metaclass__ = ABCMeta
    def __init__(self,identifier_string,graph_files_folder,graph_file_string,data_file_string,time_index):
        self.name_string = "CryptoCurrencies"
        self.identifier_string = identifier_string
        self.graph_file_string = graph_file_string
        self.graph_files_folder = graph_files_folder
        self.data_file_string = data_file_string
        self.time_index = time_index 
        
        FromFileGraph.__init__(self,self.name_string,identifier_string,graph_files_folder,graph_file_string)
        
        self.networkx_graph = nx.read_edgelist(self.graph_files_folder+graph_file_string.format(self.time_index))
        ALL_DATA = cPickle.load(open(self.graph_files_folder+data_file_string.format(time_index),"r"))
        
        self.node_data = {}
        for node_index, node_d in enumerate(ALL_DATA):
            self.node_data[node_index] = {'cryptocurrency':node_d['cryptocurrency'],
                                          'DATAFRAME_FIT':node_d['DATAFRAME_FIT'],
                                          'AV_DATEOFCRASH':node_d['AV_DATEOFCRASH'],
                                          'STD_DATEOFCRASH':node_d[ 'STD_DATEOFCRASH']}
            
    def ecosystem_crash(self):
        self.ecosystem_crash = {}
        for node_index, node_d in self.node_data.iteritems():
            self.ecosystem_crash[node_d['cryptocurrency']] = node_d['AV_DATEOFCRASH']
        return self.ecosystem_crash
    
#==============================================================
#                           FINITE PROCESS 
#==============================================================
    
class FiniteProcessGraphs(BayesianGraph):
    """
    This class is a superclass for all types of kernels (positive definite functions).
    """
    __metaclass__ = ABCMeta
    types_of_network = {1:"Undirected",2:"Directed",3:"Bipartite"}
    
    def __init__(self, identifier_string,randomMeasure,network=None,type_of_network=1):
        self.name_string = "FiniteProcessGraph"
        self.measure = randomMeasure
        self.type_of_network = type_of_network
        self.identifier_string = identifier_string
        
        BayesianGraph.__init__(self,self.name_string,self.identifier_string)
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

class CaronFoxGraphs(BayesianGraph):
    """
    This class is a superclass for all types of kernels (positive definite functions).
    """
    __metaclass__ = ABCMeta
    types_of_network = {1:"Undirected",2:"Directed",3:"Bipartite"}
    
    def __init__(self, identifier_string,randomMeasure,network=None,type_of_network=1):
        self.name_string = "CaronFox"
        self.type_of_network = type_of_network
        
        #obtain measure values
        self.tau = randomMeasure.tau
        self.alpha = randomMeasure.alpha
        self.sigma = randomMeasure.sigma
        
        self.measure = randomMeasure
        self.identifier_string = identifier_string
         
        BayesianGraph.__init__(self,self.name_string,self.identifier_string)
        if network != None:
            print "Network Given"
            self.inferMeasure()
        else:
            self.network = self.generateNetwork()
            
    def generateNetwork(self,sigma_increment=0.,tau_increment=0.,table_and_costumers=None):
        """
        We generate the network according to the paper:
        
        Sparse Graphs Using Exchangable Random Measures
        Francois Caron, Emily B. Fox 
        
        """
        self.network = nx.Graph()
        self.full_graph_measure = gamma.rvs(self.sigma+sigma_increment,self.tau+tau_increment) 
        self.number_of_arrivals =  poisson.rvs(self.full_graph_measure**2.) # THIS CORRESPONDS TO d_t^*

        costumer_seats,Thetas,numberOfSeatedCostumers = self.measure.normalized_random_measure(self.number_of_arrivals*2,
                                                                                               table_and_costumers)
        for k in range(self.number_of_arrivals):
            Uk1 = costumer_seats[2*k]
            Uk2 = costumer_seats[2*k+1]
            try:
                w = self.network.edge[Uk1][Uk2]["weight"]
                self.network.add_edge(Uk1,Uk2,weight=w+1)
            except:
                self.network.add_edge(Uk1,Uk2,weight=1)

        return self.network
    #===============================
    # INFERENCE
    #===============================
    def inferMeasures(self):
        return None
    
    