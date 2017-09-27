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

from graph_dynamics.networks.datatypes import PerraGraph

class TxGraph(PerraGraph):
        
        def __init__(self, numberOfNodes, activity_gamma, rescaling_factor, threshold_min, delta_t, number_walkers, amount_pareto_gama, amount_threshold ):
            """
              Constructor

              Parameters

                int     numberOfNodes:     to specify number of nodes of the graph
                float   activity_gamma:    alpha value of pareto distribution formula
                int     rescaling_factor:  number which scales the activity potential
                float   threshold_min:     set the min values of the activity potential from the pareto distribution
                int     delta_t:           time gap

            """
            PerraGraph.__init__(self, numberOfNodes, activity_gamma, rescaling_factor, threshold_min, delta_t, number_walkers)
            ######################### config variables #########################
            
            self.init_amount(amount_pareto_gama, amount_threshold)

        ######################### PRIVATE  METHODS  #########################


        # ===============================================
        # WALKERS FUNCTIONS
        # ===============================================

        def transfer_walker(self, _from,_to):
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
            A = pareto.rvs(amount_pareto_gama, loc=amount_threshold, scale=1, size=self.networkx_graph.number_of_nodes())
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