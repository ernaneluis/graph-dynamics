'''
Created on Apr 17, 2017

@author: cesar
'''


import unittest
import networkx as nx
import matplotlib.pyplot as plt


from graph_dynamics.dynamics import PittWalker
from graph_dynamics.random_measures import process
from graph_dynamics.utils import graph_paths_visualization
from graph_dynamics.networks.datatypes import CaronFoxGraphs


class Test(unittest.TestCase):


    def evaluateEdgesMemory(self):
        #Defines process for the graph ########################################

        self.tau = 1.
        self.alpha = 20.
        self.sigma = self.alpha
        self.process_identifier_string = "GammaProcess"
        G = process.GammaProcess(self.process_identifier_string,
                                 self.sigma,
                                 self.tau,
                                 self.alpha,
                                 K=100)
        self.graph_identifier = "CaronFoxTest"
        self.CaronFoxGraph = CaronFoxGraphs(self.graph_identifier,G)

        #Defines Dynamics ######################################################
        self.phi = 1.
        self.rho = 100.
        Palla = PittWalker.PallaDynamics(self.phi,self.rho,self.CaronFoxGraph)
        # generate dynamics
        Palla.generateNetworkPaths(3)

    def generateHiddenPaths(self):

        #Defines process for the graph ########################################
        self.tau = 1.
        self.alpha = 20.
        self.sigma = self.alpha
        self.process_identifier = "GammaProcess"
        G = process.GammaProcess(self.process_identifier,
                                 self.sigma,
                                 self.tau,
                                 self.alpha,
                                 K=100)
        self.graph_identifier = "CaronFoxTest"
        self.CaronFoxGraph = CaronFoxGraphs(self.graph_identifier,G)

        #Defines Dynamics ######################################################
        self.phi = 1.
        self.rho = 0.
        number_of_steps = 5
        number_of_steps_in_memory = 1

        simulations_directory = "/Users/ernaneluis/Developer/graph-dynamics/simulations/"
        #gd_directory = "/home/cesar/Desktop/Simulations/"
        gd_dynamical_parameters = {"number_of_steps":number_of_steps,
                                   "number_of_steps_in_memory":number_of_steps_in_memory,
                                   "simulations_directory":simulations_directory,
                                   "dynamics_identifier":"palladynamic2embeddings",
                                   "graph_class":"CaronFox",
                                   "verbose":True,
                                   "datetime_timeseries":False,
                                   "initial_date":1}

        #Macro States ========================================================================
        nargs = {"input":"../../data/graph/karate.edgelist",
                "dimensions":128,
                 "directed":False,
                 "p":0.001,
                 "q":2,
                 "num_walks":10,
                 "walk_length":80,
                 "window_size":10,
                 "workers":8,
                 "iter":1,
                 "weighted":False,
                 "undirected":True,
                 "output":"../../data/emb/karate.emb"}

        bigclam_nargs = {
                    "max_number_of_iterations": 100,
                    "error": 0.001,
                    "beta": 0.001
                }

        gd_dynamical_parameters["macrostates"] =   [
                                                    ("basic_stats",()),
                                                    ("bigclam",(bigclam_nargs,))
                                                   ]

                                                   #("node2vec_macrostates",(nargs,))]

        Palla = PittWalker.PallaDynamics(self.phi,
                                         self.rho,
                                         self.CaronFoxGraph,
                                         gd_dynamical_parameters)

        # generate dynamics
        Palla.evolve(number_of_steps,self.CaronFoxGraph)
        #graph_paths_visualization.plotGraphPaths(graph_paths, "palla_dynamics")

if __name__ == '__main__':
    import sys;sys.argv = ['','Test.generateHiddenPaths']
    unittest.main()