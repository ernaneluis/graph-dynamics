'''
Created on Jun 23, 2017

@author: cesar
'''

import unittest
import networkx as nx
import matplotlib.pyplot as plt


from tag2hierarchy.hierarchy import tree2Dict
from tag2hierarchy.hierarchy import treeHandlers

from graph_dynamics.networks.communities import HierarchicalMixedMembership


class Test(unittest.TestCase):
    

    def generateHierarchicalMixedMembership(self):
        dictTree = {"name":"A","children":[{"name":"B","children":None},
                                   {"name":"C","children":[{"name":"D","children":None},{"name":"E","children":None},{"name":"F","children":None}]}]}
        hierarchy = tree2Dict.fromDictTreeToObjectTree([dictTree])
        treeHandlers.setBranch(hierarchy)
        
        numberOfNodes = 1000
        backgroundProbability = 0.99
        inheritanceProbability = 0.8
        dirichlet_prior = 0.7
        
        HMM = HierarchicalMixedMembership(numberOfNodes,
                                            hierarchy,
                                            backgroundProbability,
                                            inheritanceProbability,
                                            dirichlet_prior)
        
        adjancecy_matrix = HMM.get_adjancency_matrix()
        P = HMM.get_probabilities()
        plt.imshow(adjancecy_matrix)
        plt.show()
        
        
if __name__ == '__main__':
    import sys;sys.argv = ['','Test.generateHierarchicalMixedMembership']
    unittest.main()