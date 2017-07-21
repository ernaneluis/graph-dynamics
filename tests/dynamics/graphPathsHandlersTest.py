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
from graph_dynamics.dynamics import GraphsFormatsHandlers

class Test(unittest.TestCase):

    def evaluateGraphPathHandlers(self):
        """
        """
        #directory = "/home/cesar/Data/Dynamics/EmpiricalData/"
        #temporalFileName = 'sx-mathoverflow.txt'
        #dynamics_identifier = 'sx-mathoverflow'

        directory = "/home/cesar/Desktop/Doctorado/Projects/Networks/Embeddings/Data/"
        temporalFileName = 'sx-mathoverflow.txt'
        dynamics_identifier = 'sx-mathoverflow'
        gd_folder = "/home/cesar/Desktop/Doctorado/Projects/Networks/Dynamics/Simulations/sx-mathoverflow_gd/"
        GraphsFormatsHandlers.seriesFromTemporalGraph(gd_folder,
                                                   dynamics_identifier,
                                                   directory+temporalFileName,
                                                   cumulative=True,
                                                   stepsInGraph="months",
                                                   parseunix=True)

if __name__ == '__main__':
    import sys;sys.argv = ['','Test.evaluateGraphPathHandlers']
    unittest.main()
