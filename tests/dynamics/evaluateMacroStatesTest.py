'''
Created on Jul 18, 2017

@author: cesar
'''

import sys
sys.path.append("../../")

import json
import unittest
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("../")
from graph_dynamics.dynamics import Macrostates

class Test(unittest.TestCase):
    
    def evaluateMacrostatesTest(self):
        """
        """
        nargs = ({"input":"../../data/graph/karate.edgelist",
                "dimensions":128,
                 "directed":False,
                 "p":0.001,
                 "q":2,
                 "num_walks":10,
                 "walk_length":80,
                 "window_size":10,
                 "workers":8,
                 "iter":1,
                 "weighted":False,
                 "undirected":True,
                 "output":"../../data/emb/karate.emb"},)
         
        #macrostates_names = [("pagerank",())]
        macrostates_names = [("basic_stats",())]
        #macrostates_names = []
        #gd_directory = "/home/cesar/Desktop/Doctorado/Projects/Networks/Dynamics/Simulations/cit-HepPh_gd/"
        gd_directory = "/home/cesar/Desktop/Doctorado/Projects/Networks/Dynamics/Simulations/palladynamic2embeddings_gd/"
        gd_directory = "/home/cesar/Desktop/Doctorado/Projects/Networks/Dynamics/Simulations/CommunityForestFire_gd/"
        macrostates_run_ideintifier = "basics"
        Macrostates.evaluate_vanilla_macrostates(gd_directory, 
                                                 macrostates_names, 
                                                 macrostates_run_ideintifier)    
        
    def windowTimeSeriesTest(self):
        
        gd_directory = "/home/cesar/Desktop/Doctorado/Projects/Networks/Dynamics/Simulations/palladynamic2embeddings_gd/"
        macrostates_run_ideintifier = "newnodes" 
        macrostates_names  = [("new_nodes",())]
        window = 1
        rolling = True
        Macrostates.evaluate_vanilla_macrostates_window(gd_directory, macrostates_names, macrostates_run_ideintifier, window, rolling)
        
if __name__ == '__main__':
    import sys;sys.argv = ['','Test.evaluateMacrostatesTest','Test.windowTimeSeriesTest']
    unittest.main()
