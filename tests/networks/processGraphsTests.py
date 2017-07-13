'''
Created on May 3, 2017

@author: cesar
'''
import json
import unittest
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from graph_dynamics.random_measures import process
from graph_dynamics.networks.datatypes import CaronFoxGraphs

class Test(unittest.TestCase):
    
    def generateCaronFox(self):
        
        self.alpha = 30.
        self.tau = 1.
        self.sigma = self.alpha
        self.process_identifier_string = "GammaProcess"

        G = process.GammaProcess(self.process_identifier_string,
                                 self.sigma,
                                 self.tau,
                                 self.alpha,
                                 K=100)
        
        G.plotProcess()
        self.graph_identifier = "CaronFoxTest"
        CaronFoxGraph = CaronFoxGraphs(self.graph_identifier,G)
        nx.draw(CaronFoxGraph.networkx_graph)
        plt.show()
        
if __name__ == '__main__':
    import sys;sys.argv = ['','Test.generateCaronFox']
    unittest.main()