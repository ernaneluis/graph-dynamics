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
from graph_dynamics.networks.datatypes import VanillaGraph

#matplotlib.rcParams['ps.useafm'] = True
#matplotlib.rcParams['text.usetex'] = True
#matplotlib.rcParams['pdf.use14corefonts'] = True

class Test(unittest.TestCase):
    
    def forestFireTest(self):
        initial_graph = nx.barabasi_albert_graph(100,3)
        number_of_steps = 85
                
        BurnExpFireP = False
        StartNNodes = 1
        ForwBurnProb = 0.2
        BackBurnProb = 0.32
        DecayProb = 1.0
        Take2AmbasPrb =  0.
        OrphanPrb =  0.

        forestFireParameters = (BurnExpFireP,StartNNodes,ForwBurnProb,BackBurnProb,DecayProb,Take2AmbasPrb,OrphanPrb) 
        timeSeriesOfNodes = np.ones(number_of_steps)*10
        number_of_steps_in_memory = 5
         
        gd_directory = "/home/cesar/Desktop/Doctorado/Projects/Networks/Dynamics/Simulations/"
        #gd_directory = "/home/cesar/Desktop/Simulations/"
        DYNAMICS_PARAMETERS = {"number_of_steps":number_of_steps,
                                "number_of_steps_in_memory":number_of_steps_in_memory,
                                "simulations_directory":gd_directory,
                                "dynamics_identifier":"ForestFireK",
                                "macrostates":[("degree_distribution",())],
                                "graph_class":"VanillaGraph",
                                "datetime_timeseries":False,
                                "initial_date":1,
                                "verbose":True}
        vanilla_graph = VanillaGraph("Vanilla", 
                                     graph_state={"None":None}, 
                                     networkx_graph=initial_graph)
        ForestFireDynamics = GenerativeDynamics.ForestFire(vanilla_graph, 
                                                           forestFireParameters,
                                                           timeSeriesOfNodes,
                                                           DYNAMICS_PARAMETERS)        
        ForestFireDynamics.evolve(40, vanilla_graph)
        graph_paths = ForestFireDynamics.get_graph_path_window(0, 20)
        nx_graph_paths = [g.get_networkx() for g in graph_paths]
        fig, ax = plt.subplots(1,1,figsize=(24,12))
        graph_paths_visualization.plotGraphPaths(ax,nx_graph_paths, "forest_fire_{0}")
        
if __name__ == '__main__':
    import sys;sys.argv = ['','Test.forestFireTest']
    unittest.main()