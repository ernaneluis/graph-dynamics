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
         
        gd_dynamics = "/home/cesar/Desktop/Doctorado/Projects/Networks/Dynamics/Simulations/Empy-States_gd/"
        #macrostates_names = [("node2vec_macrostates",(nargs,)),("basic_stats")]
        macrostates_names = [("basic_stats",())]
        #macrostates_names = [("pagerank",())]
        macrostates_run_ideintifier = "basicstatsMacro"
        
        Macrostates.evaluate_vanilla_macrostates(gd_dynamics, macrostates_names, macrostates_run_ideintifier)    
        
if __name__ == '__main__':
    import sys;sys.argv = ['','Test.evaluateMacrostatesTest']
    unittest.main()
