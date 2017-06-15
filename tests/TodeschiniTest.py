'''
Created on Jun 14, 2017

@author: cesar
'''

import sys
sys.path.append("../")

import json
import unittest
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from graph_dynamics.random_measures import process

from graph_dynamics.networks.communities import CommunityTodeschiniCaronGraph

class Test(unittest.TestCase):
    
    def generateTodeschiniGraph(self):
        
        self.identifier_string = "GammaProcessTest"
        self.alpha = 20.
        self.sigma = self.alpha
        self.tau = 1000000000.

        G = process.GammaProcess(self.identifier_string,
                                 self.sigma,
                                 self.tau,
                                 self.alpha,
                                 K=200)
        
        numberOfCommunities =  3
        bK = np.ones(numberOfCommunities)*0.000001
        #bK[:int(numberOfCommunities*0.5)] = bK[:int(numberOfCommunities*0.5)]*
        aK = np.ones(numberOfCommunities)*0.01
        #aK[:int(numberOfCommunities*0.5)] = aK[:int(numberOfCommunities*0.5)]*20 
        gammaK = np. ones(numberOfCommunities)*0.000001
        #gammaK[:int(numberOfCommunities*0.5)] = gammaK[:int(numberOfCommunities*0.5)]*2 
        
        T = CommunityTodeschiniCaronGraph("Test",
                                          numberOfCommunities,
                                          bK,aK,gammaK,G)
        
        print T.AffiliationMatrix
        nx.draw(T.get_networkx())
        plt.show()
        
if __name__ == '__main__':
    import sys;sys.argv = ['','Test.generateTodeschiniGraph']
    unittest.main()