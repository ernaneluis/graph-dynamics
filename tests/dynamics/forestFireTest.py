'''
Created on Jul 3, 2017

@author: cesar
'''

import sys
sys.path.append("../../")

import json
import unittest
import matplotlib
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from graph_dynamics.dynamics import GenerativeDynamics
from graph_dynamics.utils import graph_paths_visualization
from graph_dynamics.dynamics import GraphsFormatsHandlers

matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True

class Test(unittest.TestCase):
    
    def forestFireTest(self):
        barabasi_graph = nx.barabasi_albert_graph(100,3)
        number_of_steps = 100
                
        BurnExpFireP = False
        StartNNodes = 1
        ForwBurnProb = 0.2
        BackBurnProb = 0.32
        DecayProb = 1.0
        Take2AmbasPrb =  0.
        OrphanPrb =  0.

        forestFireParameters = (BurnExpFireP,StartNNodes,ForwBurnProb,BackBurnProb,DecayProb,Take2AmbasPrb,OrphanPrb) 
        timeSeriesOfNodes = np.ones(number_of_steps)*10
        number_of_steps_in_memory = 10
         
        gd_directory = "/home/cesar/Desktop/Doctorado/Projects/Networks/Dynamics/Simulations/"
        #gd_directory = "/home/cesar/Desktop/Simulations/"
        gd_dynamical_parameters = {"number_of_steps":number_of_steps,
                                   "number_of_steps_in_memory":number_of_steps_in_memory,
                                   "gd_directory":gd_directory,
                                   "dynamics_identifier":"forestfire",
                                   "macrostates":["degree_distribution"],
                                   "graph_class":"VanillaGraph",
                                   "verbose":True}
        
        dynamics = GenerativeDynamics.ForestFire(barabasi_graph, 
                                                 forestFireParameters,
                                                 timeSeriesOfNodes)
        
        graph_paths = dynamics.generate_graphs_paths(10)
        #static_graph = GraphsFormatsHandlers.staticGraphInducedBySeries(graph_paths)
        #temporal_graph = GraphsFormatsHandlers.temporalGraphFromSeries(graph_paths)
        graph_paths_visualization.plotGraphPaths(graph_paths, "forest_fire")
        
if __name__ == '__main__':
    import sys;sys.argv = ['','Test.forestFireTest']
    unittest.main()