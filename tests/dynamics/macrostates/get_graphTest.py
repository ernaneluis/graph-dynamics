'''
Created on Sep 1, 2017

@author: cesar
'''
import operator
import unittest
from matplotlib import pyplot as plt
from graph_dynamics import dynamics
from graph_dynamics.utils import gd_files_handler
from graph_dynamics.dynamics import MacrostatesHandlers
from graph_dynamics.dynamics import datatypes

class Test(unittest.TestCase):

    def timeSeriesTest(self):
        #gd_directory = "/home/cesar/Desktop/Doctorado/Projects/Networks/Dynamics/Simulations/cit-HepPh_gd/"
        #gd_directory = "/home/cesar/Desktop/GraphsDynamics/Simulations/palladynamic_gd/"
        gd_directory = "/home/cesar/Desktop/Doctorado/Projects/Networks/Dynamics/Simulations/CommunityForestFire3_gd/"
        gd_directory = "/home/cesar/Desktop/Doctorado/Projects/Networks/Embeddings/Communities_gd/"
        ALL_TIME_INDEXES,DYNAMICS_PARAMETERS,macroNumbers = gd_files_handler.gd_folder_stats(gd_directory,True)
        
        graph_object = datatypes.get_graph_from_dynamics(gd_directory,0)
        print graph_object.get_graph_state()



if __name__ == '__main__':
    import sys;sys.argv = ['','Test.timeSeriesTest']
    unittest.main()