'''
Created on Apr 17, 2017

@author: cesar
'''


import unittest
import networkx as nx
import matplotlib.pyplot as plt


from bayesian_networks.dynamics import PittWalker
from bayesian_networks.random_measures import process
from bayesian_networks.networks.datatypes import CaronFoxGraphs


class Test(unittest.TestCase):
    

    def generateHiddenPaths(self):
        
        self.tau = 1.
        self.alpha = 20.
        self.sigma = self.alpha
        self.process_identifier_string = "GammaProcess"

        G = process.GammaProcess(self.process_identifier_string,
                                 self.sigma,
                                 self.tau,
                                 self.alpha,
                                 K=100)
        G.plotProcess()
        self.graph_identifier = "CaronFoxTest"
        self.CaronFoxGraph = CaronFoxGraphs(self.graph_identifier,G)
        nx.draw(self.CaronFoxGraph.network)
        plt.show()
        
        #Dynamics 
        self.phi = 1.
        self.rho = 1.
        
        Palla = PittWalker.PallaDynamics(self.phi,self.rho,self.CaronFoxGraph)
        #C_TimeSeries = Palla.generateHiddenPath(10)
        Networks_TimeSeries = Palla.generateNetworkPaths(10)
        
if __name__ == '__main__':
    import sys;sys.argv = ['','Test.generateHiddenPaths']
    unittest.main()