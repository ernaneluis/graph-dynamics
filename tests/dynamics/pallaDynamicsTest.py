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
        Palla.generateNetworkPaths_1(3)
       
    def generateHiddenPaths(self):
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
        self.phi = 0.2
        self.rho = 1.
        Palla = PittWalker.PallaDynamics(self.phi,self.rho,self.CaronFoxGraph)
        # generate dynamics
        graph_paths = Palla.generateNetworkPaths_1(10)
        graph_paths_visualization.plotGraphPaths(graph_paths, "palla_dynamics")
        
if __name__ == '__main__':
    import sys;sys.argv = ['','Test.generateHiddenPaths']
    unittest.main()