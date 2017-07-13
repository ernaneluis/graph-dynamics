'''
Created on Jul 10, 2017

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
from graph_dynamics.dynamics import FromFilesDynamics

matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True

class Test(unittest.TestCase):
    
    def cryptocurrencyEcosystemDynamicsTest(self):
        identifier_string = "Crash Detections (Data up to 24/06/17)"
        graph_files_folder = "/home/cesar/Desktop/Doctorado/Projects/Networks/Cryptocurrencies/Results/cryptocurrencies_graphdynamics/"
        data_file_string = 'cryptocurrencies_graphdynamics_snapshot_{0}_data.cpickle3'
        graph_file_string = 'cryptocurrencies_graphdynamics_snapshot_{0}_graph.txt'
        
        EcosystemDynamics = FromFilesDynamics.CryptoCurrencyEcosystemDynamics(identifier_string,
                                                                              graph_files_folder,
                                                                              graph_file_string,
                                                                              data_file_string)
        
        self.graph_paths = EcosystemDynamics.generate_graphs_paths()
        fig, ax = plt.subplots(1,1,figsize=(24,12))
        graph_paths_visualization.plotGraphPaths(ax,self.graph_paths,series_name=EcosystemDynamics.identifier_string+" {0}")
        
if __name__ == '__main__':
    import sys;sys.argv = ['','Test.cryptocurrencyEcosystemDynamicsTest']
    unittest.main()