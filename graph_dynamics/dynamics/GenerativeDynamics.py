'''
Created on Jun 9, 2017

@author: cesar
'''
from scipy.stats import pareto, norm, bernoulli
import sys
import snap
import copy
import random
import numpy as np
import networkx as nx

from graph_dynamics.utils import snap_handlers
from graph_dynamics.networks.datatypes import VanillaGraph
from graph_dynamics.dynamics.datatypes import GraphsDynamics

#==========================================================================
# FOREST FIRE
#==========================================================================

class ForestFire(GraphsDynamics):
    """
    This is a wrapper for the snap function Forest Fire
    
    """
    def __init__(self, initial_graph,forestFireParameters,timeSeriesOfNodes,DYNAMICAL_PARAMETERS):
        """
        initial_graph: networkx graph
        
        BurnExpFireP: bool
        
        StartNNodes: int
        
        ForwBurnProb: double
        
        BackBurnProb: double
        
        DecayProb: double
        
        Take2AmbasPrb: double
        
        OrphanPrb: double
        
        timeSeriesOfNodes: numpy array
            the number of new nodes per time step
        """
        (BurnExpFireP,StartNNodes,ForwBurnProb,BackBurnProb,DecayProb,Take2AmbasPrb,OrphanPrb) = forestFireParameters
        self.BurnExpFireP = BurnExpFireP
        self.StartNNodes = StartNNodes
        self.ForwBurnProb = ForwBurnProb
        self.BackBurnProb = BackBurnProb
        self.DecayProb = DecayProb
        self.Take2AmbasPrb = Take2AmbasPrb
        self.OrphanPrb = OrphanPrb
        
        type_of_dynamics = "snap_shot"
        self.dynamics_identifier = "ForestFire"
        self.timeSeriesOfNodes = timeSeriesOfNodes
        
        DYNAMICAL_PARAMETERS["DynamicsClassParameters"]={"ForestFire":forestFireParameters,
                                                         "TimeSeriesOfNodes":list(timeSeriesOfNodes)} 
        
        
        self.DYNAMICAL_PARAMETERS = DYNAMICAL_PARAMETERS
        self.Vanilla_0 =  VanillaGraph(self.dynamics_identifier,
                                       {"None":None},
                                       initial_graph)

        self.ff = snap.TFfGGen(BurnExpFireP,StartNNodes,ForwBurnProb,
                               BackBurnProb,DecayProb,Take2AmbasPrb,OrphanPrb)
        
         
        GraphsDynamics.__init__(self, DYNAMICAL_PARAMETERS)
        
    def generate_graphs_paths(self,initial_graph,T):
        T = T - 1 
        print "initital generate paths ",initial_graph.get_networkx().number_of_nodes()
        initial_graph_nx = initial_graph.get_networkx()
        str_int = dict(zip(initial_graph_nx.nodes(),map(int,initial_graph_nx.nodes())))
        initial_graph_nx = nx.relabel_nodes(initial_graph_nx, str_int) 
        snap_graph = snap_handlers.nx_to_snap(initial_graph_nx)
        
        graph_series = [VanillaGraph(self.dynamics_identifier,
                                     {"None":None},
                                     snap_handlers.snap_to_nx(snap_graph))]
        numberOfNodes = graph_series[0].get_networkx().number_of_nodes()
        
        try:
            for i in range(0,int(T)):
                number_of_new_nodes = self.timeSeriesOfNodes[self.latest_index+i]
                numberOfNodes += number_of_new_nodes
                self.ff.SetGraph(snap_graph)
                self.ff.AddNodes(int(numberOfNodes), True)
                graph_series.append(VanillaGraph(self.dynamics_identifier,
                                                 {"None":None},
                                                 snap_handlers.snap_to_nx(snap_graph)))
        except:
            print sys.exc_info()
            print "Number of steps for series not match nodes time series"
            raise Exception
        
        return graph_series
    
    def set_graph_path(self):
        """
        Empirical Data
        """
        return None
        
    def inference_on_graphs_paths(self):
        """
        Learning/Training
        """
        return None
        
    def get_dynamics_state(self):
        """
        """
        return self.DYNAMICAL_PARAMETERS
 

#==========================================================================
# Activity Driven DYNAMICS
#==========================================================================

class ActivityDrivenDynamics(GraphsDynamics):
    def __init__(self, initial_graph, DYNAMICAL_PARAMETERS, extra_parameters):
        """
          Constructor

          Parameters

            TxGraph initial_graph:            initial state of a TxGraph instance
            int     number_of_connections:    max number of connections/edges a node can do

        """


        DYNAMICAL_PARAMETERS["DynamicsClassParameters"] = {"ActivityDrivenDynamics": None}
        # graph is a type of Acitivy Driven or Perra Graph
        self.initial_graph = initial_graph
        self.number_of_connections = extra_parameters["number_of_connections"]
        self.DYNAMICAL_PARAMETERS = DYNAMICAL_PARAMETERS
        self.extra_parameters = extra_parameters

        GraphsDynamics.__init__(self, DYNAMICAL_PARAMETERS)
        # ==================  set up the initial graph  ====================================================

        self.activity_potential = self.__calculateActivityPotential(extra_parameters["activity_gamma"],
                                                                    extra_parameters["threshold_min"],
                                                                    extra_parameters["number_of_nodes"])
        # creating list of nodes with index from 0 de N-1  adding to the graph
        self.initial_graph.get_networkx().add_nodes_from(list(xrange(extra_parameters["number_of_nodes"])))

        # run over all nodes to set initial attributes
        for n in self.initial_graph.get_networkx().nodes():
            ## what is the purpose of rescaling factor?
            # ai = xi*n => probability per unit time to create new interactions with other nodes
            # activity_firing_rate is an probability number than [0,1]
            self.initial_graph.get_networkx().node[n]['activity_firing_rate'] = self.activity_potential[n] * extra_parameters["rescaling_factor"]
            # With probability ai*delta_t each vertex i becomes active and generates m links that are connected to m other randomly selected vertices
            self.initial_graph.get_networkx().node[n]['activity_probability'] = self.initial_graph.get_networkx().node[n]['activity_firing_rate'] * extra_parameters["delta_t"]



    # Abstract methods ====================================================

    def generate_graphs_paths(self, initial_graph, number_of_steps):
        """
          Method

          Parameters

            int     number_of_steps:   Total time steps to perform dynamics
            string  output_type:

        """
        graph_series = [self.initial_graph]
        for T in range(1, number_of_steps):
            graph_series.append(self.evolve_function(graph_series[T - 1]))

        return graph_series

    def set_graph_path(self):
        """
        Empirical Data
        """
        raise None

    def inference_on_graphs_paths(self, graphs_paths, output_type, dynamical_process=None):
        """
        Learning/Training
        """
        return None

    def get_dynamics_state(self):
        return self.DYNAMICAL_PARAMETERS

    def evolve_function(self, dynamical_process=None):
        """
        """

        # 0 clear connections
        self.initial_graph.get_networkx().remove_edges_from(self.initial_graph.get_networkx().edges())
        # 1 select nodes to be active
        after_connections = self.__set_nodes_active()
        # 2 make conenctions from activacted nodes
        before_connections = self.__set_connections()

        # TODO: perra dynamics will handle the walker case
        # 3 make random walk
        # walked = self.__set_propagate_walker()

        return copy.deepcopy(self.initial_graph)

    # Class methods ====================================================

    def __calculateActivityPotential(self, activity_gamma, threshold_min, number_of_nodes):
        ## calculating the activity potential following pareto distribution
        X = pareto.rvs(activity_gamma, loc=threshold_min,size=number_of_nodes)  # get N samples from  pareto distribution
        X = X / max(X)  # every one smaller than one
        return np.take(X, np.where(X > threshold_min)[0])  # using the thershold

    def __set_nodes_active(self):
        for n in self.initial_graph.get_networkx().nodes():
            self.initial_graph.set_node_type(n)

        return self.initial_graph

    def __set_connections(self):
        # list of choosed active nodes
        active_nodes = self.initial_graph.get_active_nodes()
        # for each selected node make M connections
        for node in active_nodes:
            # select random M nodes to make M connection
            selected_nodes = [(node, random.randint(0, self.initial_graph.get_number_of_nodes() - 1)) for e in
                              range(self.number_of_connections)]
            # make connections/edges
            self.initial_graph.get_networkx().add_edges_from(selected_nodes)

        return self.initial_graph

    # def __set_propagate_walker(self):
    #     walkers = self.initial_graph.get_walkers()
    #     for node in walkers:
    #         # look at their neighbors: nodes that the walker is making an connection
    #         neighbors_nodes = self.initial_graph.get_networkx().neighbors(node)
    #
    #         if len(neighbors_nodes) > 0:
    #             # when a walker will not propagate he will stay at the same node
    #             neighbors_nodes.append(node)
    #
    #             selected_neighbor = np.random.choice(neighbors_nodes, size=1, replace=False)
    #             selected_neighbor = selected_neighbor[0]
    #
    #             self.initial_graph.transfer_walker(_from=node, _to=selected_neighbor)
    #
    #             if node != selected_neighbor:
    #                 print("walker  #" + str(node) + " moved to node #" + str(selected_neighbor))
    #             else:
    #                 print("walker node #" + str(node) + " did not move ")
    #         else:
    #             print("walker  #" + str(node) + " is trap, cant move because there is no node to go(edge)")
    #
    #     return self.initial_graph


# ==========================================================================
# BITCOIN DYNAMICS
# ==========================================================================

# class TxDynamics(PerraDynamics):
#     def __init__(self, initial_graph, number_of_connections):
#         """
#           Constructor
#
#           Parameters
#
#             TxGraph initial_graph:            initial state of a TxGraph instance
#             int     number_of_connections:    max number of connections/edges a node can do
#
#         """
#
#         name_string = "GammaProcess"
#         type_of_dynamics = "SnapShot"
#         # GraphsDynamics.__init__(self, initial_graph, type_of_dynamics, number_of_connections)
#
#         # #graph is a type of TxGraph
#         # self.GRAPH = initial_graph
#         # self.number_of_connections = number_of_connections
#
#         PerraDynamics.__init__(self, initial_graph, number_of_connections)
#
#
#
#
#         # def evolve_function(self,dynamical_process=None):
#         #     """
#         #     """
#         #     #0 clear connections
#         #     self.GRAPH.networkx_graph.remove_edges_from(self.GRAPH.networkx_graph.edges())
#         #     #1 select nodes to be active
#         #     after_connections = self.__set_nodes_active()
#         #     #2 make conenctions from activacted nodes
#         #     before_connections = self.__set_connections()
#         #     #3 make random walk
#         #     #if amount were set then graph can do random walk
#         #     walked = self.__set_propagate_walker()
#         #
#         #
#         #     return copy.deepcopy(self.GRAPH)
#         #

