'''
Created on Jul 25, 2017

@author: cesar
'''


import sys
sys.path.append("../../")


import unittest
sys.path.append("../")

import json
import numpy as np
import networkx as nx
from graph_dynamics.embeddings import deep_walk, alignment
from graph_dynamics.embeddings import utils
from gensim.models import Word2Vec
import matplotlib.pyplot as plt

from sklearn import preprocessing

class Test(unittest.TestCase):
    
    def basicAlignment(self):
        """
        """
        G = nx.karate_club_graph()
        walks = deep_walk.deepWalk(G, 10)
        
        walksA = np.random.permutation(walks)
        walksB = np.random.permutation(walks)
    
        modelA = Word2Vec(walksA.tolist(), size=2, window=10, min_count=0, sg=1, workers=8, iter=50)
        modelB = Word2Vec(walksB.tolist(), size=2, window=10, min_count=0, sg=1, workers=8, iter=50)
    
        w_a = np.asarray([modelA[str(node)] for node in G.nodes()])
        w_b = np.asarray([modelB[str(node)] for node in G.nodes()])
        
        w_a_aligned, w_b_aligned =  alignment.procrustes_align(w_a, w_b)
        alignment.plot_w(w_a_aligned, w_b_aligned)
        plt.show()
        
if __name__ == '__main__':
    import sys;sys.argv = ['','Test.basicAlignment']
    unittest.main()