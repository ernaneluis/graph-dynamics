'''
Created on May 3, 2017

@author: cesar
'''


import copy
import cPickle
import numpy as np
from networkx import nx
from networkx.readwrite import json_graph
from numpy import cumsum, sort, sum, searchsorted
from numpy.random import rand
from numpy.random import choice

import time
import random
import math

from scipy.stats import poisson, gamma
from abc import ABCMeta, abstractmethod
from scipy.stats import pareto, norm, bernoulli
from _pyio import __metaclass__

from graph_dynamics.random_measures.process import GammaProcess


#matplotlib.rcParams['ps.useafm'] = True
#matplotlib.rcParams['text.usetex'] = True
#matplotlib.rcParams['pdf.use14corefonts'] = True

class Graph(object):##<------------------
    """
    This is the main class for the graphs dynamics library
    it defines the minimum methods required for the abstract
    handling of the dynamic
    """    
    __metaclass__ = ABCMeta

    def __init__(self,name_string,identifier_string,graph_state):
        """
        Abstract constructor
        
        Parameters
        ----------
        name_string: string
            this is the name of the class defined
        identifier_string: string 
            this is the name of the particular graph object
        graph_state: string 
            this is a json with all the information required to define the graph
            if the user provides the adjancency matrix and this json, the graph should be completly defined
            this is required in order to save thesimulations in thime, and be able to recover at any point in
            time 
        """
        self.name_string = name_string
        self.identifier_string = identifier_string
        self.graph_state = graph_state
        
    @abstractmethod
    def get_graph_state(self):
        """
        This function should return a json object with all 
        parameters required to initialize such a graph 
        """
        raise NotImplemented()
        
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

class VanillaGraph(Graph):
    """
    This graph can be used as a handler in order to
    analyse files, it simply holds a networkx graph
    the state is an empty json
    """
    def __init__(self,identifier_string=None,graph_state=None,networkx_graph=None):
        self.name_string = "VanillaGraph"
        self.type_of_network = 1
        if identifier_string == None:
            try:
                self.identifier_string = graph_state["graph_identifier"]
            except:
                self.identifier_string = "Vanilla"
        else:
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

    def get_graph_state(self):
        """
        This function should return a json object with all 
        parameters required to initialize such a graph 
        """
        return self.graph_state
            
    def get_networkx(self):
        return self.networkx_graph
          
    def get_adjancency_matrix(self):
        return nx.adjacency_matrix(self.networkx_graph)

    def get_edge_list(self):
        return self.networkx_graph.edge
     
    def get_number_of_edges(self):
        return self.networkx_graph.number_of_edges()
     
    def get_number_of_nodes(self):
        return self.networkx_graph.number_of_nodes()     
    
#=====================================================
# COMMUNITY
#=====================================================

 
    
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
#                    FINITE PROCESS 
#==============================================================

class FiniteProcessGraphs(BayesianGraph):
    """
    Using the finite approximation from a gamma process
    a graph is created in the CaronFox sense
    
    Parameters
    ----------
    
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

    #===============================
    # LIBRARY GRAPH REQUIREMENTS
    #===============================
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
    
    def get_graph_state(self):
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
        
    def get_graph_state(self):
        return self.graph_state
        
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
    
    #===============================
    # LIBRARY GRAPH REQUIREMENTS
    #===============================
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

#======================================================


#==============================================================
#                           ACTIVITY DRIVEN GRAPH CLASS
#==============================================================

class ActivityDrivenGraph(VanillaGraph):
    def __init__(self, graph_state, networkx_graph):
        """
        Abstract constructor

        Parameters
        ----------
        name_string: string
            this is the name of the class defined
        identifier_string: string
            this is the name of the particular graph object
        graph_state: string
            this is a json with all the information required to define the graph
            if the user provides the adjancency matrix and this json, the graph should be completly defined
            this is required in order to save thesimulations in thime, and be able to recover at any point in
            time



        int     number_of_nodes:     to specify number of nodes of the graph
        float   activity_gamma:    alpha value of pareto distribution formula
        int     rescaling_factor:  number which scales the activity potential
        float   threshold_min:     set the min values of the activity potential from the pareto distribution
        int     delta_t:           time gap
        """

        ######################### config variables #########################
        self.name_string = "ActivityDrivenGraph"
        self.activity_potential = []
        VanillaGraph.__init__(self,self.name_string,graph_state,networkx_graph)



    ######################### PUBLIC  METHODS  #########################

    def set_activity_node(self, nonde_id, activity_potential_n):

        activity_firing_rate = activity_potential_n * self.activity_rescaling_factor
        activity_probability = activity_firing_rate * self.activity_delta_t

        self.get_networkx().node[nonde_id]['activity_firing_rate'] = activity_firing_rate
        # With probability ai*delta_t each vertex i becomes active and generates m links that are connected to m other randomly selected vertices
        self.get_networkx().node[nonde_id]['activity_probability'] = activity_probability

    def set_activity(self, activity_potential, activity_rescaling_factor, activity_delta_t):
        self.activity_potential = activity_potential
        self.activity_rescaling_factor = activity_rescaling_factor
        self.activity_delta_t = activity_delta_t
        # graph_state = self

        # run over all nodes to set initial attributes
        for n, node in enumerate(self.get_networkx().nodes()):
            self.set_activity_node(n, activity_potential[n])

            ## what is the purpose of rescaling factor?
            # ai = xi*n => probability per unit time to create new interactions with other nodes
            # activity_firing_rate is an probability number than [0,1]

            # s = sum(activity_potential)

            # activity_firing_rate = activity_potential[n] * activity_rescaling_factor
            #
            # activity_probability = activity_firing_rate * activity_delta_t
            #
            # self.get_networkx().node[n]['activity_firing_rate'] = activity_firing_rate
            # # With probability ai*delta_t each vertex i becomes active and generates m links that are connected to m other randomly selected vertices
            # self.get_networkx().node[n]['activity_probability'] = activity_probability

        # return self

    def set_nodes_active(self):
        for n in self.networkx_graph.nodes():
            self.set_node_type(n)

    def get_active_nodes(self):
        # return the list of choosed active nodes
        return [n for n in self.networkx_graph.nodes() if self.networkx_graph.node[n]['type'] == 1]

    def get_activity_probability(self, node):
        return self.networkx_graph.node[node]['activity_probability']

    def get_activity_firing_rate(self, node):
        return self.networkx_graph.node[node]['activity_firing_rate']

    def get_node_type(self, node):
        return self.networkx_graph.node[node]['type']

    def set_node_type(self, node):
        ## assign a node attribute nonactive or active
        # is sample the activity once and do the bernoully sample at each time step
        activity_probability = self.get_activity_probability(node)
        if (activity_probability > 1):
            activity_probability = 1 / activity_probability
        # set if a node is active or not
        isActive = bernoulli.rvs(activity_probability)
        self.networkx_graph.node[node]['type'] = isActive


    def add_new_node(self, new_node_id, new_node_amount, new_node_activity_firing_rate, type):

        attributes = {'amount': new_node_amount,
                      'activity_firing_rate': 0,
                      'activity_probability': 0,
                      'type': 0}

        self.networkx_graph.add_node(new_node_id, attr_dict=attributes)
        # it will site node type from acitivity
        self.set_node_type(new_node_id)
        # it will set activity_firing_rate and activity_probability
        self.set_activity_node(new_node_id, new_node_activity_firing_rate)


class BitcoinGraph(ActivityDrivenGraph):
    def __init__(self, graph_state, networkx_graph):


        ######################### config variables #########################
        ActivityDrivenGraph.__init__(self, graph_state, networkx_graph)


    # ==================================== INIT FUNCTIONS ====================================

    def init_amount(self, size, amount_pareto_gama, amount_threshold):
        # calculate the wealth distribution following pareto law
        A = pareto.rvs(amount_pareto_gama, loc=amount_threshold, scale=1, size=size)
        # A = A / max(A)
        self.set_amount(A)

    def init_activity_potential(self, size, activity_gamma, threshold_min, activity_rescaling_factor, activity_delta_t):
        ## calculating the activity potential following pareto distribution
        X = pareto.rvs(activity_gamma, loc=threshold_min, size=size)  # get N samples from  pareto distribution
        # X = X / max(X)  # every one smaller than one
        # return np.take(X, np.where(X > threshold_min)[0])  # using the thershold

        # X = self.softmax(X)
        self.set_activity(X, activity_rescaling_factor, activity_delta_t)
        # set_activity(self, activity_potential, activity_rescaling_factor, activity_delta_t):
        return X



    # ==================================== ACTIVITY FUNCTIONS ====================================

    def transfer_function(self, x):
        # log sigmoid
        # return  1/( 1 +pow(math.e,(-1*x)) ) : model 2
        # return x/(1+abs(x)) model 5 0.32
        return x / math.sqrt(1 + pow(x, 2))

    def recalculate_activity_potential(self):
        # run over all nodes to set initial attributes
        for n, node in enumerate(self.networkx_graph.nodes()):
            money_i_t   = self.networkx_graph.node[n]['amount']
            alpha       = self.get_activity_firing_rate(n)
            activity_i  = self.transfer_function(alpha * money_i_t)

            self.activity_potential[n] = activity_i
            self.set_activity_node(n, activity_i)

        # activity_potential = activity_potential / max(activity_potential)
        # activity_potential = self.softmax(activity_potential)
        # return graph_state

        # TODO a_i(t) = log(alpha * Money_i(t))
        # TODO 1/1-pow(e, (-a*Money_i(t)) )


    # ==================================== AMOUNT FUNCTIONS ====================================

    def set_amount(self, amount):
        self.amount = amount
        # run over all nodes to set initial attributes
        for n in self.networkx_graph.nodes():
            # setting the node the initial amount of wealth
            self.networkx_graph.node[n]['amount'] = amount[n]

    def get_amount(self, node):
        return self.networkx_graph.node[node]['amount']

    def add_amount(self, node, amount):
        self.networkx_graph.node[node]['amount'] += amount

    def remove_amount(self, node, amount):
        self.networkx_graph.node[node]['amount'] -= amount

    def get_nodes_amounts(self):
        return [self.networkx_graph.node[node]['amount'] for node in self.networkx_graph.nodes()]

    def transfer_amount(self, _from, _to, amount_to_move):
        # right now when a walker move, he moves all the money
        # TODO: add a probability to move % of the amount
        # amount_to_move = self.get_amount(_from)
        self.remove_amount(_from, amount_to_move)
        self.add_amount(_to, amount_to_move)
        return amount_to_move

    # ==================================== MEMORY FUNCTIONS ====================================

    def memory_append(self, from_node, to_node):
        # save on node 'to' who send the money
        hasMemory = 'memory' in self.networkx_graph.node[to_node]

        if(hasMemory==False):
            self.networkx_graph.node[to_node]['memory'] = list()


        memory = self.networkx_graph.node[to_node]['memory']
        if(len(memory) == 5):
            memory.pop(0)

        memory.append(from_node)

        self.networkx_graph.node[to_node]['memory'] = memory

    def get_memory(self, node):
        hasMemory = 'memory' in self.networkx_graph.node[node]

        if (hasMemory == False):
            self.networkx_graph.node[node]['memory'] = list()

        return self.networkx_graph.node[node]['memory']

    # ==================================== OTHER FUNCTIONS ====================================

    def update_graph_state(self):
        gstate = json_graph.node_link_data(self.networkx_graph)
        gstate = gstate["nodes"]
        self.graph_state = gstate

    def calculate_amount(self, from_node, number_of_connections):
        amount      = self.get_amount(from_node)
        amount      = amount/number_of_connections
        return amount

    def set_connections(self, number_of_connections, delta_in_seconds):

        # list of choosed active nodes
        active_nodes = self.get_active_nodes()
        # for each selected node make M connections
        for node in active_nodes:
            # 3-tuples (u,v,d) for an edge attribute dict d, or
            # select random M nodes to make M connection
            from_node    = node
            memory_nodes = self.get_memory(from_node)
            n_nodes      = self.get_number_of_nodes()
            elements     = self.networkx_graph.nodes()
            weights      = np.array(self.activity_potential[:n_nodes])

            if len(memory_nodes) > 0:
                w = weights[memory_nodes]
                w = w * 20
                weights[memory_nodes] = w

            weights_p = self.softmax(weights)

            to_list = choice(elements, p=weights_p, size=number_of_connections, replace=True)

            selected_nodes = []
            for idx, to_node in enumerate(to_list):
                # to_node     = random.randint(0, graph_state.get_number_of_nodes() - 1)
                amount = self.calculate_amount(from_node, number_of_connections)

                self.transfer_amount(from_node, to_node, amount)

                data_node = {
                    'time': random.randint(int(time.time()), int(time.time()) + delta_in_seconds),
                    'amount': amount
                }
                edge = (from_node, to_node, data_node)
                selected_nodes.append(edge)

                self.memory_append(from_node, to_node)

            # the connections are made as bucket and in our case each time connection step is a day in real life
            # we must simulate a day of connections by timestamp

            self.networkx_graph.add_edges_from(selected_nodes)

    def add_new_nodes(self, number_new_nodes):
        new_nodes = []
        for i in range(number_new_nodes):
            max_node_id = max(self.networkx_graph.nodes())

            new_node_id = max_node_id + 1
            new_node_amount = self.amount[new_node_id]
            new_node_activity_firing_rate = self.activity_potential[new_node_id]
            new_node_type = 0  # not active

            self.add_new_node(new_node_id, new_node_amount, new_node_activity_firing_rate, new_node_type)

            new_nodes.append(new_node_id)
        return new_nodes



     # https://en.m.wikipedia.org/wiki/Softmax_function

    def softmax(self, X):
        z_exp = [math.exp(i) for i in X]
        sum_z_exp = sum(z_exp)
        softmax = [i / sum_z_exp for i in z_exp]
        s = sum(softmax)
        return softmax

class BitcoinMemoryGraph(BitcoinGraph):

    def __init__(self, graph_state, networkx_graph):
        ######################### config variables #########################
        BitcoinGraph.__init__(self, graph_state, networkx_graph)

    def init_memory_activity_potential(self, size, activity_gamma, threshold_min, activity_rescaling_factor, activity_delta_t):
        ## calculating the activity potential following pareto distribution
        X = pareto.rvs(activity_gamma, loc=threshold_min, size=size)  # get N samples from  pareto distribution
        self.set_memory_activity(X, activity_rescaling_factor, activity_delta_t)
        # set_activity(self, activity_potential, activity_rescaling_factor, activity_delta_t):
        return X

    def set_memory_activity(self, activity_potential, activity_rescaling_factor, activity_delta_t):
        self.memory_activity_potential = activity_potential
        self.memory_activity_rescaling_factor = activity_rescaling_factor
        self.memory_activity_delta_t = activity_delta_t
        # graph_state = self

        # run over all nodes to set initial attributes
        for n, node in enumerate(self.get_networkx().nodes()):
            self.set_memory_activity_node(n, activity_potential[n])

    def set_memory_activity_node(self, nonde_id, activity_potential_n):
        activity_firing_rate = activity_potential_n * self.memory_activity_rescaling_factor
        activity_probability = activity_firing_rate * self.memory_activity_delta_t

        self.get_networkx().node[nonde_id]['memory_activity_firing_rate'] = activity_firing_rate
        # With probability ai*delta_t each vertex i becomes active and generates m links that are connected to m other randomly selected vertices
        self.get_networkx().node[nonde_id]['memory_activity_probability'] = activity_probability

    def set_nodes_memory_active(self):
        for n in self.networkx_graph.nodes():
            self.set_node_memory_type(n)

    def set_node_memory_type(self, node):
        ## assign a node attribute nonactive or active
        # is sample the activity once and do the bernoully sample at each time step
        activity_probability = self.get_memory_activity_probability(node)
        if (activity_probability > 1):
            activity_probability = 1 / activity_probability
        # set if a node is active or not
        isActive = bernoulli.rvs(activity_probability)
        self.networkx_graph.node[node]['memory_type'] = isActive

    def set_memory_connections(self, memory_number_of_connections, delta_in_seconds):

        # list of choosed active nodes
        active_nodes = self.get_memory_active_nodes()
        # for each selected node make M connections
        for node in active_nodes:
            # 3-tuples (u,v,d) for an edge attribute dict d, or
            # select random M nodes to make M connection
            from_node       = node
            memory_nodes    = self.get_memory(from_node)

            if len(memory_nodes) > 0:

                to_list = choice(memory_nodes, size=memory_number_of_connections)

                selected_nodes = []
                for idx, to_node in enumerate(to_list):

                    amount = self.calculate_amount(from_node, memory_number_of_connections)

                    self.transfer_amount(from_node, to_node, amount)

                    data_node = {
                        'time': random.randint(int(time.time()), int(time.time()) + delta_in_seconds),
                        'amount': amount
                    }
                    edge = (from_node, to_node, data_node)
                    selected_nodes.append(edge)

                    # self.memory_append(from_node, to_node)

                # the connections are made as bucket and in our case each time connection step is a day in real life
                # we must simulate a day of connections by timestamp

                self.networkx_graph.add_edges_from(selected_nodes)

    def add_new_memory_nodes(self, number_new_nodes):

        for idx, new_node_id in enumerate(number_new_nodes):

            new_node_memory_activity_firing_rate = self.memory_activity_potential[new_node_id]

            self.networkx_graph.node[new_node_id]['memory_type'] = 0

            self.set_memory_activity_node(new_node_id, new_node_memory_activity_firing_rate)

            self.set_node_memory_type(new_node_id)


    def recalculate_memory_activity_potential(self):
        a = 1
        # run over all nodes to set initial attributes
        # for n, node in enumerate(self.networkx_graph.nodes()):
        #     money_i_t   = self.networkx_graph.node[n]['amount']
        #     alpha       = self.get_activity_firing_rate(n)
        #     activity_i  = self.transfer_function(alpha * money_i_t)
        #
        #     self.activity_potential[n] = activity_i
        #     self.set_activity_node(n, activity_i)


    def get_memory_active_nodes(self):
        # return the list of choosed active nodes
        return [n for n in self.networkx_graph.nodes() if self.networkx_graph.node[n]['memory_type'] == 1]

    def get_memory_activity_probability(self, node):
        return self.networkx_graph.node[node]['memory_activity_probability']

    def get_memory_activity_firing_rate(self, node):
        return self.networkx_graph.node[node]['memory_activity_firing_rate']

    def get_node_memory_type(self, node):
        return self.networkx_graph.node[node]['memory_type']



#==============================================================
#                           PERRA GRAPH CLASS: HAS WALKERS
#==============================================================

class PerraGraph(ActivityDrivenGraph):
    def __init__(self, graph_state, networkx_graph):


        ######################### config variables #########################
        ActivityDrivenGraph.__init__(self, graph_state, networkx_graph)
        # self, identifier_string, graph_state, networkx_graph, number_of_nodes, activity_gamma, rescaling_factor, threshold_min, delta_t)


    # ===============================================
    # WALKERS FUNCTIONS
    # ===============================================
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

graph_class_dictionary = {
    "CaronFox":CaronFoxGraphs,
    "VanillaGraph":VanillaGraph,
    "ActivityDrivenGraph": ActivityDrivenGraph,
    "PerraGraph": PerraGraph,
    "BitcoinGraph": BitcoinGraph,
    "BitcoinMemoryGraph": BitcoinMemoryGraph

}