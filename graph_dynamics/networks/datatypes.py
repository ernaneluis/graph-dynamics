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
from scipy.stats import pareto, norm, bernoulli
from _pyio import __metaclass__

from graph_dynamics.random_measures.process import GammaProcess

matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True

class Graph(object):##<------------------
    """
    """    
    __metaclass__ = ABCMeta

    def __init__(self,name_string,identifier_string,graph_state):
        self.name_string        = name_string
        self.identifier_string  = identifier_string
        self.graph_state        = graph_state
        self.networkx_graph     = nx.Graph()

    def get_graph_state(self):
        """
                This function should return a json object with all
                parameters required to initialize such a graph
        """
        return self.graph_state

    def get_networkx(self):
        return self.networkx_graph

    def get_adjancency_matrix(self):
        return nx.adj_matrix(self.networkx_graph)

    def get_edge_list(self):
        return self.networkx_graph.edges()

    def get_number_of_edges(self):
        return self.networkx_graph.number_of_edges()

    def get_number_of_nodes(self):
        return self.networkx_graph.number_of_nodes()


class VanillaGraph(Graph):
    """
    This graph can be used as a handler in order to
    analyse files, it simply holds a networkx graph
    """
    def __init__(self,identifier_string,graph_state=None,networkx_graph=None):
        self.name_string = "VanillaGraph"
        self.type_of_network = 1
        self.identifier_string = identifier_string
        #initialize with parameters
        if networkx_graph==None:
            self.networkx_graph = nx.barabasi_albert_graph(100, 3)
            self.graph_state = {"None":None}
        #initialize with json object
        else:
            self.graph_state = copy.copy(graph_state)
            self.networkx_graph = networkx_graph
            
        Graph.__init__(self,self.name_string,self.identifier_string,self.graph_state)


    
class CommunityGraphs(Graph):
    __metaclass__ = ABCMeta

    def __init__(self,name_string,identifier_string):
        self.name_string = name_string
        self.identifier_string = identifier_string

    @abstractmethod
    def get_nodes(self):
        raise NotImplemented()
    
#==============================================================
#                           BAYESIAN CLASS 
#==============================================================

class BayesianGraph(Graph):
    """
    This class is a class for graphs which require random 
    measures or 
    """
    __metaclass__ = ABCMeta

    def __init__(self,name_string,identifier_string,graph_state):
        self.name_string = name_string
        self.identifier_string = identifier_string
        Graph.__init__(self,name_string, identifier_string, graph_state)

    @abstractmethod
    def inferMeasures(self):
        raise NotImplemented()

    @abstractmethod
    def generateNetwork(self):
        raise NotImplemented()

#==============================================================
#                           RECSYS
#==============================================================

class OwnershipGraph(Graph):
    """
    This class is a superclass for bipartite graph define as
    costumer-product relationships, these class is intended 
    for recommender systems
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

    @abstractmethod
    def projectBinaryMatrix(self):
        raise NotImplemented()

#==============================================================
#                           FINITE PROCESS 
#==============================================================

class FiniteProcessGraphs(BayesianGraph):
    """
    This class is a superclass for all types of kernels (positive definite functions).
    """
    types_of_network = {1:"Undirected",2:"Directed",3:"Bipartite"}
    # TODO: finite process graphs has to be initiated with json object
    def __init__(self,identifier_string,randomMeasure,network=None,type_of_network=1):
        self.name_string = "FiniteProcessGraph"
        self.measure = randomMeasure
        self.type_of_network = type_of_network
        self.identifier_string = identifier_string
                
        if network != None:
            print "Network Given"
            self.inferMeasure()
        else:
            self.networkx_graph = self.generateNetwork()
            
        self.graph_state = {"identifier_string":self.identifier_string,
                            "type_of_network":self.type_of_network,
                            "number_of_nodes":self.get_number_of_nodes(),
                            "number_of_edges":self.get_number_of_edges(),
                            "measure_state":self.measure.get_measure_state()}

        BayesianGraph.__init__(self,self.name_string,self.identifier_string,self.graph_state)
    
        print "#============================"
        print "#  Bayesian Network Ready    "
        print "# Number of nodes {0}        ".format(self.networkx_graph.number_of_nodes())
        print "# Number of edges {0}        ".format(self.networkx_graph.number_of_edges())
        print "#============================"
        
    def generateNetwork(self):
        """
        We generate the networkx_graph according to the paper:
        
        Sparse Graphs Using Exchangable Random Measures
        Francois Caron, Emily B. Fox 
        
        But Instead of Using the Normalized Completly Random Measure Approach, we directly 
        use the Weights as defined by the Finite Measure  
        """        
        #THE NETWORKX GRAPH FOR PLOTTING AND REFERENCE
        self.networkx_graph = nx.Graph()
        for node_i in range(int(self.measure.K)):
            for node_j in range(int(self.measure.K)):
                if node_i != node_j:
                    n_ij = poisson.rvs(2*self.measure.W[node_i]*self.measure.W[node_j])
                    if n_ij > 0:
                        if self.type_of_network == 1:
                            self.networkx_graph.add_edge(node_i,node_j)
                        if self.type_of_network == 2:
                            self.networkx_graph.add_edge(node_i,node_j,weight=n_ij)
                else:
                    n_ii = poisson.rvs(self.measure.W[node_i]*self.measure.W[node_j])
                    if n_ii > 0:
                        if self.type_of_network == 1:
                            self.networkx_graph.add_edge(node_i,node_j)
                        if self.type_of_network == 2:
                            self.networkx_graph.add_edge(node_i,node_j,weight=n_ii)
        return self.networkx_graph
    
    
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
    This class defines the graph from:
    
    Sparse Graphs Using Exchangable Random Measures
        Francois Caron, Emily B. Fox
         
    We can initialize the graph via parameters or pasing a json object with the following keys:
    measure_state["tau"]
    measure_state["measure_state"]: another json defining the measure 
                                    which defines the graph 
                                    
            measure_state["identifier_string"]
            measure_state["sigma"]
            measure_state["alpha"]
            measure_state["tau"]
            measure_state["measure"]["W"]
            measure_state["measure"]["Theta"]
            measure_state["lambda_measure_state"]
             
    """
    __metaclass__ = ABCMeta
    types_of_network = {1:"Undirected",
                        2:"Directed",
                        3:"Bipartite"}
    
    def __init__(self, identifier_string=None,
                       randomMeasure=None,
                       graph_state=None,networkx_graph=None):
        
        self.name_string = "CaronFox"
        self.type_of_network = 1
        
        #initialize with parameters
        if graph_state==None:
            self.tau = randomMeasure.tau
            self.alpha = randomMeasure.alpha
            self.sigma = randomMeasure.sigma
            
            self.measure = randomMeasure
            self.identifier_string = identifier_string
            self.networkx_graph = self.generateNetwork()
            self.C = None           #VARIABLE REQUIERED FOR PALLA DYNAMICS
                
            self.graph_state = {"graph_name":self.name_string,
                                "graph_identifier":self.identifier_string,
                                "measure_state":self.measure.get_measure_state(),
                                "C":self.C}
            
        #initialize with json object
        else:
            self.graph_state = copy.copy(graph_state)
            measure_state = copy.copy(graph_state["measure_state"])
            
            self.tau = measure_state["tau"]
            self.alpha = measure_state["alpha"]
            self.sigma = measure_state["sigma"]
            self.identifier_string = measure_state["identifier_string"] 
            self.measure = GammaProcess(measure_state=measure_state)
            self.networkx_graph = networkx_graph
            
        BayesianGraph.__init__(self,self.name_string,self.identifier_string,self.graph_state)
    
    def set_C(self,C):
        self.C = C
        self.graph_state["C"] = C

        
    def generateNetwork(self,sigma_increment=0.,tau_increment=0.,table_and_costumers=None):
        """
        We generate the networkx_graph according to the paper:
        
        Sparse Graphs Using Exchangable Random Measures
        Francois Caron, Emily B. Fox 
        
        """
        self.networkx_graph = nx.Graph()
        self.full_graph_measure = gamma.rvs(self.sigma+sigma_increment,self.tau+tau_increment) 
        self.number_of_arrivals =  poisson.rvs(self.full_graph_measure**2.) # THIS CORRESPONDS TO d_t^*

        costumer_seats,Thetas,numberOfSeatedCostumers = self.measure.normalized_random_measure(self.number_of_arrivals*2,
                                                                                               table_and_costumers)
        for k in range(self.number_of_arrivals):
            Uk1 = costumer_seats[2*k]
            Uk2 = costumer_seats[2*k+1]
            try:
                w = self.networkx_graph.edge[Uk1][Uk2]["weight"]
                self.networkx_graph.add_edge(Uk1,Uk2,weight=w+1)
            except:
                self.networkx_graph.add_edge(Uk1,Uk2,weight=1)

        return self.networkx_graph
    
    #===============================
    # INFERENCE
    #===============================
    def inferMeasures(self):
        return None



#==============================================================
#                           FILE
#==============================================================

class FromFileGraph(object):
    """
    This class is a superclass for all types of kernels (positive definite functions).
    """
    __metaclass__ = ABCMeta

    def __init__(self,name_string,identifier_string,graph_files_folder,graph_file_string):
        self.name_string = name_string
        self.identifier_string = identifier_string

#==============================================================
#                           FILE
#==============================================================

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

# ==============================================================
#                           ACTIVITY DRIVEN
# ==============================================================

class ActivityDrivenGraph(Graph):


    def __init__(self, identifier_string,
                 graph_state=None,
                 numberOfNodes=1,
                 activity_gamma=2.8,
                 rescaling_factor=None,
                 threshold_min=0.001,
                 delta_t=1):
        """
          Constructor

          Parameters

            int     numberOfNodes:     to specify number of nodes of the graph
            float   activity_gamma:    alpha value of pareto distribution formula
            int     rescaling_factor:  number which scales the activity potential
            float   threshold_min:     set the min values of the activity potential from the pareto distribution
            int     delta_t:           time gap

        """

        ######################### config variables #########################
        # initialize with parameters
        Graph.__init__(self, "ActivityDrivenGraph", identifier_string, copy.copy(graph_state))


        # creating graph
        self.networkx_graph     = nx.Graph()
        self.activity_potential = self.__calculateActivityPotential(activity_gamma, threshold_min, numberOfNodes)
        self.networkx_graph.add_nodes_from(list(xrange(numberOfNodes)))  # creating list of nodes with index from 0 de N-1  adding to the graph

        ######################### end config variables #########################
        self.hasGraphAmount = False

        ######################### initializing graph  #########################
        # run over all nodes to set initial attributes
        for n in self.networkx_graph.nodes():
            ## what is the purpose of rescaling factor?
            # ai = xi*n => probability per unit time to create new interactions with other nodes
            # activity_firing_rate is an probability number than [0,1]
            self.networkx_graph.node[n]['activity_firing_rate'] = self.activity_potential[n] * rescaling_factor

            # With probability ai*delta_t each vertex i becomes active and generates m links that are connected to m other randomly selected vertices
            self.networkx_graph.node[n]['activity_probability'] = self.networkx_graph.node[n]['activity_firing_rate'] * delta_t



    # graph paths should return a list of graphs objects
            ######################### PRIVATE  METHODS  #########################

    def __calculateActivityPotential(self, activity_gamma, threshold_min, numberOfNodes):
        ## calculating the activity potential following pareto distribution
        X = pareto.rvs(activity_gamma, loc=threshold_min, size=numberOfNodes)  # get N samples from  pareto distribution
        X = X / max(X)  # every one smaller than one
        return np.take(X, np.where(X > threshold_min)[0])  # using the thershold

    ######################### PUBLIC  METHODS  #########################

    def get_active_nodes(self):
        # return the list of choosed active nodes
        return [n for n in self.networkx_graph.nodes() if self.networkx_graph.node[n]['type'] == 1]

    def number_of_nodes(self):
        return self.networkx_graph.number_of_nodes()

    def get_activity_firing_rate(self, node):
        return self.networkx_graph.node[node]['activity_firing_rate']

    def get_node_type(self, node):
        return self.networkx_graph.node[node]['type']

    def set_node_type(self, node):
        ## assign a node attribute nonactive or active
        # is sample the activity once and do the bernoully sample at each time step
        activity_firing_rate = self.get_activity_firing_rate(node)
        if (activity_firing_rate > 1):
            activity_firing_rate = 1 / activity_firing_rate
        # set if a node is active or not
        self.networkx_graph.node[node]['type'] = bernoulli.rvs(activity_firing_rate)

# ==============================================================
#                           PERRA
# ==============================================================

class PerraGraph(ActivityDrivenGraph):

    def __init__(self, identifier_string,
                 graph_state,
                 numberOfNodes,
                 activity_gamma,
                 rescaling_factor,
                 threshold_min,
                 delta_t,
                 number_walkers):
        """
          Constructor

          Parameters

            int     numberOfNodes:     to specify number of nodes of the graph
            float   activity_gamma:    alpha value of pareto distribution formula
            int     rescaling_factor:  number which scales the activity potential
            float   threshold_min:     set the min values of the activity potential from the pareto distribution
            int     delta_t:           time gap

        """


        ######################### config variables #########################
        ActivityDrivenGraph.__init__(self, identifier_string, graph_state, numberOfNodes, activity_gamma, rescaling_factor, threshold_min, delta_t)

        ######################### initializing graph  #########################
        # run over all nodes to set initial attributes
        for n in self.networkx_graph.nodes():
            self.networkx_graph.node[n]['walker'] = 0

        self.init_walkers(number_walkers)

    # ===============================================
    # WALKERS FUNCTIONS
    # ===============================================

    def init_walkers(self, number_walkers):
        # create W walkers on those nodes
        self.number_walkers = number_walkers
        generate_walkers = np.random.choice(self.networkx_graph.nodes(), size=number_walkers, replace=False)
        for node in generate_walkers:
            self.add_walker(node)

    def transfer_walker(self, _from, _to):
        self.remove_walker(_from)
        self.add_walker(_to)

    def add_walker(self, node):
        self.networkx_graph.node[node]['walker'] += 1

    def remove_walker(self, node):
        self.networkx_graph.node[node]['walker'] -= 1

    # return list of nodes which has walkers
    def get_walkers(self):
        out = []
        for node in self.networkx_graph.nodes():
            if self.networkx_graph.node[node]['walker'] > 0:
                out += [node]

        return out

# ==============================================================
#                 Bitcoin Transactions Graph
# ==============================================================

class TxGraph(PerraGraph):

    def __init__(self, identifier_string,
                 graph_state,
                 numberOfNodes,
                 activity_gamma,
                 rescaling_factor,
                 threshold_min,
                 delta_t,
                 number_walkers,
                 amount_pareto_gama,
                 amount_threshold):
        """
          Constructor

          Parameters

            int     numberOfNodes:     to specify number of nodes of the graph
            float   activity_gamma:    alpha value of pareto distribution formula
            int     rescaling_factor:  number which scales the activity potential
            float   threshold_min:     set the min values of the activity potential from the pareto distribution
            int     delta_t:           time gap

        """
        PerraGraph.__init__(self, identifier_string,
                            graph_state,
                            numberOfNodes,
                            activity_gamma,
                            rescaling_factor,
                            threshold_min,
                            delta_t,
                            number_walkers)
        ######################### config variables #########################

        self.init_amount(amount_pareto_gama, amount_threshold)

    ######################### PRIVATE  METHODS  #########################


    # ===============================================
    # WALKERS FUNCTIONS
    # ===============================================

    def transfer_walker(self, _from, _to):
        self.remove_walker(_from)
        self.add_walker(_to)

        # right now when a walker move, he moves all the money
        # TODO: add a probability to move % of the amount
        amount_to_move = self.get_amount(_from)
        self.remove_amount(_from, amount_to_move)
        self.add_amount(_to, amount_to_move)

    # ===============================================
    # AMOUNT FUNCTIONS
    # ===============================================

    # setting the node the initial amount of wealth
    def init_amount(self, amount_pareto_gama, amount_threshold):
        self.hasGraphAmount = True

        # calculate the wealth distribution following pareto law
        A = pareto.rvs(amount_pareto_gama, loc=amount_threshold, scale=1,
                       size=self.networkx_graph.number_of_nodes())
        self.amount = A / max(A)

        # run over all nodes to set initial attributes
        for n in self.networkx_graph.nodes():
            # setting the node the initial amount of wealth
            self.networkx_graph.node[n]['amount'] = self.amount[n]

    def get_amount(self, node):
        return self.networkx_graph.node[node]['amount']

    def add_amount(self, node, amount):
        self.networkx_graph.node[node]['amount'] += amount

    def remove_amount(self, node, amount):
        self.networkx_graph.node[node]['amount'] -= amount

    def get_nodes_amounts(self):
        return [self.networkx_graph.node[node]['amount'] for node in self.networkx_graph.nodes()]

#======================================================

graph_class_dictionary = {"CaronFox":CaronFoxGraphs}