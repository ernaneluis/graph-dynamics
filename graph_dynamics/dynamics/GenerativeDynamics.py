'''
Created on Jun 9, 2017

@author: cesar
'''
import snap
import math
import time
import pylab
import random
import numpy as np
import networkx as nx
from scipy import stats
from scipy.stats import pareto, norm, bernoulli
import copy

from graph_dynamics.dynamics.datatypes import GraphsDynamics
from graph_dynamics.networks.tx_graph import TxGraph
from graph_dynamics.utils import snap_handlers

#==========================================================================
# FOREST FIRE
#==========================================================================

class ForestFire(GraphsDynamics):
    """
    This is a wrapper for the snap function Forest Fire
    
    """
    def __init__(self, initial_graph,
                 BurnExpFireP,StartNNodes,ForwBurnProb,
                 BackBurnProb,DecayProb,Take2AmbasPrb,OrphanPrb,timeSeriesOfNodes):
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
        type_of_dynamics = "snap_shot"
        GraphsDynamics.__init__(self, initial_graph, type_of_dynamics, None)
        self.initial_graph = initial_graph
        
        self.ff = snap.TFfGGen(BurnExpFireP,StartNNodes,ForwBurnProb,
                          BackBurnProb,DecayProb,Take2AmbasPrb,OrphanPrb)
        
        self.BurnExpFireP = BurnExpFireP
        self.StartNNodes = StartNNodes
        self.ForwBurnProb = ForwBurnProb
        self.BackBurnProb = BackBurnProb
        self.DecayProb = DecayProb
        self.Take2AmbasPrb = Take2AmbasPrb
        self.OrphanPrb = OrphanPrb
        
        self.timeSeriesOfNodes = timeSeriesOfNodes 
    
    def generate_graphs_paths(self,number_of_steps,output_type=list):
        snap_graph = snap_handlers.nx_to_snap(self.initial_graph)
        graph_series = [snap_handlers.snap_to_nx(snap_graph)]
        numberOfNodes = graph_series[0].number_of_nodes()
        
        if number_of_steps == len(self.timeSeriesOfNodes):
            if output_type == list:
                for j, number_of_new_nodes in enumerate(self.timeSeriesOfNodes):
                    numberOfNodes += number_of_new_nodes
                    self.ff.SetGraph(snap_graph)
                    self.ff.AddNodes(int(numberOfNodes), True)
                    graph_series.append(snap_handlers.snap_to_nx(snap_graph))
        else:
            print "Number of steps for series not match nodes time series"
            raise Exception
        
        return graph_series
    
    def inference_on_graphs_paths(self,graphs_paths,output_type,dynamical_process):
        return None



#==========================================================================
# Perra DYNAMICS
#==========================================================================

class PerraDynamics(GraphsDynamics):
    def __init__(self, initial_graph, number_of_connections):
        """
          Constructor

          Parameters

            TxGraph initial_graph:            initial state of a TxGraph instance
            int     number_of_connections:    max number of connections/edges a node can do

        """

        name_string = "GammaProcess"
        type_of_dynamics = "SnapShot"
        GraphsDynamics.__init__(self, initial_graph, type_of_dynamics, number_of_connections)

        # graph is a type of TxGraph
        self.GRAPH = initial_graph
        self.number_of_connections = number_of_connections

    def generate_graphs_paths(self, number_of_steps, output_type):
        """
          Method

          Parameters

            int     number_of_steps:   Total time steps to perform dynamics
            string  output_type:

        """

        graph_series = [self.GRAPH]
        for T in range(1, number_of_steps):
            graph_series.append(self.evolve_function(graph_series[T - 1]))

        return graph_series

    def evolve_function(self, dynamical_process=None):
        """
        """
        # 0 clear connections
        self.GRAPH.networkx_graph.remove_edges_from(self.GRAPH.networkx_graph.edges())
        # 1 select nodes to be active
        after_connections = self.__set_nodes_active()
        # 2 make conenctions from activacted nodes
        before_connections = self.__set_connections()
        # 3 make random walk
        walked = self.__set_propagate_walker()

        return copy.deepcopy(self.GRAPH)

    def inference_on_graphs_paths(self, graphs_paths, output_type, dynamical_process=None):
        """
        """
        return None

    def __set_nodes_active(self):
        for n in self.GRAPH.networkx_graph.nodes():
            self.GRAPH.set_node_type(n)

        return self.GRAPH

    def __set_connections(self):
        # list of choosed active nodes
        active_nodes = self.GRAPH.get_active_nodes()
        # for each selected node make M connections
        for node in active_nodes:
            # select random M nodes to make M connection
            selected_nodes = [(node, random.randint(0, self.GRAPH.number_of_nodes() - 1)) for e in
                              range(self.number_of_connections)]
            # make connections/edges
            self.GRAPH.networkx_graph.add_edges_from(selected_nodes)

        return self.GRAPH

    def __set_propagate_walker(self):
        walkers = self.GRAPH.get_walkers()
        for node in walkers:
            # look at their neighbors: nodes that the walker is making an connection
            neighbors_nodes = self.GRAPH.networkx_graph.neighbors(node)

            if len(neighbors_nodes) > 0:
                # when a walker will not propagate he will stay at the same node
                neighbors_nodes.append(node)

                selected_neighbor = np.random.choice(neighbors_nodes, size=1, replace=False)
                selected_neighbor = selected_neighbor[0]

                self.GRAPH.transfer_walker(_from=node, _to=selected_neighbor)

                if node != selected_neighbor:
                    print("walker  #" + str(node) + " moved to node #" + str(selected_neighbor))
                else:
                    print("walker node #" + str(node) + " did not move ")
            else:
                print("walker  #" + str(node) + " is trap, cant move because there is no node to go(edge)")

        return self.GRAPH


# ==========================================================================
# BITCOIN DYNAMICS
# ==========================================================================

class TxDynamics(PerraDynamics):
    def __init__(self, initial_graph, number_of_connections):
        """
          Constructor

          Parameters

            TxGraph initial_graph:            initial state of a TxGraph instance
            int     number_of_connections:    max number of connections/edges a node can do

        """

        name_string = "GammaProcess"
        type_of_dynamics = "SnapShot"
        # GraphsDynamics.__init__(self, initial_graph, type_of_dynamics, number_of_connections)

        # #graph is a type of TxGraph
        # self.GRAPH = initial_graph
        # self.number_of_connections = number_of_connections

        PerraDynamics.__init__(self, initial_graph, number_of_connections)




        # def evolve_function(self,dynamical_process=None):
        #     """
        #     """
        #     #0 clear connections
        #     self.GRAPH.networkx_graph.remove_edges_from(self.GRAPH.networkx_graph.edges())
        #     #1 select nodes to be active
        #     after_connections = self.__set_nodes_active()
        #     #2 make conenctions from activacted nodes
        #     before_connections = self.__set_connections()
        #     #3 make random walk
        #     #if amount were set then graph can do random walk
        #     walked = self.__set_propagate_walker()
        #
        #
        #     return copy.deepcopy(self.GRAPH)
        #

