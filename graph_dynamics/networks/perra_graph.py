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

class ActivityDrivenGraph():
    def __init__(self, numberOfNodes, activity_gamma, rescaling_factor, threshold_min, delta_t):
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

        # creating graph
        self.hasGraphAmount = False
        self.network = nx.Graph()
        self.activity_potential = self.__calculateActivityPotential(activity_gamma, threshold_min, numberOfNodes)
        self.network.add_nodes_from(list(xrange(numberOfNodes)))  # creating list of nodes with index from 0 de N-1  adding to the graph

        ######################### end config variables #########################


        ######################### initializing graph  #########################
        # run over all nodes to set initial attributes
        for n in self.network.nodes():
            ## what is the purpose of rescaling factor?
            # ai = xi*n => probability per unit time to create new interactions with other nodes
            # activity_firing_rate is an probability number than [0,1]
            self.network.node[n]['activity_firing_rate'] = self.activity_potential[n] * rescaling_factor

            # With probability ai*delta_t each vertex i becomes active and generates m links that are connected to m other randomly selected vertices
            self.network.node[n]['activity_probability'] = self.network.node[n]['activity_firing_rate'] * delta_t

    ######################### PRIVATE  METHODS  #########################

    def __calculateActivityPotential(self, activity_gamma, threshold_min, numberOfNodes):
        ## calculating the activity potential following pareto distribution
        X = pareto.rvs(activity_gamma, loc=threshold_min, size=numberOfNodes)  # get N samples from  pareto distribution
        X = X / max(X)  # every one smaller than one
        return np.take(X, np.where(X > threshold_min)[0])  # using the thershold

    ######################### PUBLIC  METHODS  #########################

    def get_active_nodes(self):
        # return the list of choosed active nodes
        return [n for n in self.network.nodes() if self.network.node[n]['type'] == 1]

    def number_of_nodes(self):
        return self.network.number_of_nodes()

    def get_activity_firing_rate(self, node):
        return self.network.node[node]['activity_firing_rate']

    def get_node_type(self, node):
        return self.network.node[node]['type']

    def set_node_type(self, node):
        ## assign a node attribute nonactive or active
        # is sample the activity once and do the bernoully sample at each time step
        activity_firing_rate = self.get_activity_firing_rate(node)
        if (activity_firing_rate > 1):
            activity_firing_rate = 1 / activity_firing_rate
        # set if a node is active or not
        self.network.node[node]['type'] = bernoulli.rvs(activity_firing_rate)


class PerraGraph(ActivityDrivenGraph):
        
        def __init__(self, numberOfNodes, activity_gamma, rescaling_factor, threshold_min, delta_t, number_walkers):
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
            ActivityDrivenGraph.__init__(self, numberOfNodes, activity_gamma, rescaling_factor, threshold_min, delta_t)


            ######################### initializing graph  #########################
            # run over all nodes to set initial attributes
            for n in self.network.nodes():
                self.network.node[n]['walker'] = 0

            self.init_walkers(number_walkers)

        # ===============================================
        # WALKERS FUNCTIONS
        # ===============================================

        def init_walkers(self, number_walkers):
            # create W walkers on those nodes
            self.number_walkers = number_walkers
            generate_walkers = np.random.choice(self.network.nodes(), size=number_walkers, replace=False)
            for node in generate_walkers:
                self.add_walker(node)

        def transfer_walker(self, _from,_to):
            self.remove_walker(_from)
            self.add_walker(_to)

        def add_walker(self, node):
            self.network.node[node]['walker'] += 1

        def remove_walker(self, node):
            self.network.node[node]['walker'] -= 1

        # return list of nodes which has walkers
        def get_walkers(self):
            out = []
            for node in self.network.nodes():
                if self.network.node[node]['walker'] > 0:
                    out += [node]

            return out

