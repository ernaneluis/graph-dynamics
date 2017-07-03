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
import copy

from graph_dynamics.dynamics.datatypes import GraphsDynamics


from graph_dynamics.networks.tx_graph import TxGraph

class TxDynamics(GraphsDynamics):
    
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

        #graph is a type of TxGraph
        self.GRAPH = initial_graph
        self.number_of_connections = number_of_connections


    def generate_graphs_series(self,number_of_steps,output_type):
        """
          Method

          Parameters

            int     number_of_steps:   Total time steps to perform dynamics
            string  output_type:

        """
        
        graph_series = [self.GRAPH]
        for T in range(1,number_of_steps):
            graph_series.append(self.evolve_function(graph_series[T-1]))
    
        return graph_series

    def evolve_function(self,dynamical_process=None):
        """
        """
        #0 clear connections
        self.GRAPH.network.remove_edges_from(self.GRAPH.network.edges())
        #1 select nodes to be active
        after_connections = self.__set_nodes_active()
        #2 make conenctions from activacted nodes
        before_connections = self.__set_connections()
        #3 make random walk
        #if amount were set then graph can do random walk
        if(self.GRAPH.hasGraphAmount):
            walked = self.__set_propagate_walker()


        return copy.deepcopy(self.GRAPH)


    def define_graphs_series(self,graphs_paths,output_type,dynamical_process=None):
        """
        """
        return None
       
    def __set_nodes_active(self):
        for n in self.GRAPH.network.nodes():
            self.GRAPH.set_node_type(n)

        return self.GRAPH

    def __set_connections(self):
        # list of choosed active nodes
        active_nodes = self.GRAPH.get_active_nodes()
        # for each selected node make M connections
        for node in active_nodes:
            # select random M nodes to make M connection
            selected_nodes = [(node, random.randint(0, self.GRAPH.number_of_nodes() - 1)) for e in range(self.number_of_connections)]
            # make connections/edges
            self.GRAPH.network.add_edges_from(selected_nodes)

        return self.GRAPH

    def __set_propagate_walker(self):
        walkers = self.GRAPH.get_walkers()
        for node in walkers:
            # look at their neighbors: nodes that the walker is making an connection
            neighbors_nodes = self.GRAPH.network.neighbors(node)

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




