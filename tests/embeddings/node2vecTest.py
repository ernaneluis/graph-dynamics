'''
Created on Jun 30, 2017

@author: cesar
'''

import sys
sys.path.append("../../")

import json
import unittest
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("../")
from graph_dynamics.embeddings import node2vec
from graph_dynamics.embeddings import utils

class Test(unittest.TestCase):
    
    def generateNode2VecEmbeddings(self):
        """
        """
        self.args = {"input":"../../data/graph/karate.edgelist",
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
                     "output":"../../data/emb/karate.emb"} 
        
        number_of_clusters = 2
        nx_G = node2vec.read_graph(self.args)
        
        G = node2vec.Graph(nx_G, 
                           self.args["directed"], 
                           self.args["p"], 
                           self.args["q"])
        
        G.preprocess_transition_probs()
        
        walks = G.simulate_walks(self.args["num_walks"], 
                                 self.args["walk_length"])
        
        node2vec.learn_embeddings(walks,self.args)
        utils.clusterEmbeddings(nx_G, self.args["output"],number_of_clusters)
        
if __name__ == '__main__':
    import sys;sys.argv = ['','Test.generateNode2VecEmbeddings']
    unittest.main()