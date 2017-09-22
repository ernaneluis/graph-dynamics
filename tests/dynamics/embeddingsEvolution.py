'''
Created on Jul 31, 2017

@author: cesar
'''
import sys
import copy
sys.path.append("../../")

import json
import unittest
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pylab
from matplotlib.pyplot import pause


from graph_dynamics.embeddings import alignment
from graph_dynamics.utils import gd_files_handler
from graph_dynamics.dynamics import MacrostatesHandlers

#matplotlib.rcParams['ps.useafm'] = True
#matplotlib.rcParams['text.usetex'] = True
#matplotlib.rcParams['pdf.use14corefonts'] = True

#matplotlib.style.use('ggplot')
#matplotlib.style.use('seaborn-talk')

colors = []
for a in plt.style.library['bmh']['axes.prop_cycle']:
    colors.append(a["color"])
    
class Test(unittest.TestCase):
    
    def embeddingsEvolutionTest(self):                                                                                                
        
        #gd_directory = "/home/cesar/Desktop/Doctorado/Projects/Networks/Dynamics/Simulations/CommunityForestFire4_gd/"
        gd_directory = "/home/cesar/Desktop/Doctorado/Projects/Networks/Embeddings/Communities_gd/"
        ALL_TIME_INDEXES,DYNAMICS_PARAMETERS,macroNumbers = gd_files_handler.gd_folder_stats(gd_directory,True)
        number_of_steps = 100
        
        node_embedding = MacrostatesHandlers.time_index_macro(gd_directory,
                 "node2vec_macrostates",
                 "aligned-node2vec",
                 0)
        
        initial_nodes = node_embedding.keys()
        new_nodes_per_time = copy.copy(alignment.all_new_nodes_per_time(gd_directory,initial_nodes,latest_index=99))
        
        #============================================================================
        # SIMULATION
        #============================================================================

        fig, ax = plt.subplots(1,1,figsize=(24,12))
        pylab.ion()
        for i in range(0,100):
            plt.xlim(-3.,3.)
            plt.ylim(-3.,3.)
            node_embedding = MacrostatesHandlers.time_index_macro(gd_directory,
                                                                  "node2vec_macrostates",
                                                                  "aligned-node2vec",
                                                                  i)
            
            w = np.array(node_embedding.values())
            w_new = np.array([node_embedding[new_node] for new_node in new_nodes_per_time[i]])
            
            plt.title("Time {0} Number of nodes {1}".format(i,len(w)))
            plt.scatter(w[:,0], w[:,1],label="{0}".format(i))
            plt.scatter(w_new[:,0], w_new[:,1],label="{0}".format(i),color="r")
            pause(0.3)
            plt.clf()
            
            
if __name__ == '__main__':
    import sys;sys.argv = ['','Test.embeddingsEvolutionTest']
    unittest.main()