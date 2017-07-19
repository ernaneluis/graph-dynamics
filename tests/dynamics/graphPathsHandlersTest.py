'''
Created on Jul 19, 2017

@author: cesar
'''
import sys
from graph_dynamics.utils.gd_files_handler import gd_folder_stats
sys.path.append("../../")

import json
import unittest
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("../")
from graph_dynamics.dynamics import GraphPathsHandlers

class Test(unittest.TestCase):
    
    def evaluateGraphPathHandlers(self):
        """
        """
        directory = "/home/cesar/Desktop/Doctorado/Projects/Networks/Embeddings/Data/"
        temporalFileName = 'Temporal-Cit-HepTh.txt'
        dynamics_identifier = 'Cit-HepTh'
        gd_folder = "/home/cesar/Desktop/Doctorado/Projects/Networks/Dynamics/Simulations/cit-HepTh_gd/"
        GraphPathsHandlers.seriesFromTemporalGraph(gd_folder, 
                                                   dynamics_identifier, 
                                                   directory+temporalFileName, 
                                                   parseunix=False)
        
if __name__ == '__main__':
    import sys;sys.argv = ['','Test.evaluateGraphPathHandlers']
    unittest.main()
