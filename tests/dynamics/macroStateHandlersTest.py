'''
Created on Jul 20, 2017

@author: cesar
'''
import unittest
from graph_dynamics.dynamics import MacrostatesHandlers
from graph_dynamics.utils import gd_files_handler
from matplotlib import pyplot as plt
import operator

class Test(unittest.TestCase):

    def timeSeriesTest(self):
        #gd_directory = "/home/cesar/Desktop/Doctorado/Projects/Networks/Dynamics/Simulations/cit-HepPh_gd/"
        #gd_directory = "/home/cesar/Desktop/GraphsDynamics/Simulations/palladynamic_gd/"
        gd_directory = "/home/cesar/Desktop/Doctorado/Projects/Networks/Dynamics/Simulations/CommunityForestFire_gd/"
        
        ALL_TIME_INDEXES,DYNAMICS_PARAMETERS,macroNumbers = gd_files_handler.gd_folder_stats(gd_directory,True)

        
        macrostate_file_indentifier = "CommunityForestFire-macros"
        macro_state_identifier = "basic_stats"
        
        macro_keys = {"number_of_nodes":"scalar"}
                      #"number_of_edges":"scalar"}

        df = MacrostatesHandlers.TS_dict_macro(gd_directory,
                                               macro_state_identifier,
                                               macrostate_file_indentifier,
                                               macro_keys)
        print df
        df.plot()
        plt.show()
        
        #macro_state_identifier = "basics_stats"
        #macrostate_file_indentifier = "basics"
        #time_index = 19

        #pagerank_result = MacrostatesHandlers.time_index_macro(gd_directory,
        #                                                            macro_state_identifier,
        #                                                            macrostate_file_indentifier,
        #                                                            time_index)
        

        #df = MacrostatesHandlers.TS_per_node_scalar_macro(gd_directory,
        #                                                  macrostate_file_indentifier,
        #                                                 macro_state_identifier,
        #                                                  None,
        #                                                  selectTop=5)
        
        #df.plot()
        #plt.show()
        
if __name__ == '__main__':
    import sys;sys.argv = ['','Test.timeSeriesTest']
    unittest.main()
