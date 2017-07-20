'''
Created on Jul 20, 2017

@author: cesar
'''
import unittest
from graph_dynamics.dynamics import MacrostatesHandlers
from graph_dynamics.utils import gd_files_handler
from matplotlib import pyplot as plt


class Test(unittest.TestCase):
    
    def timeSeriesTest(self):
        gd_directory = "/home/cesar/Desktop/Doctorado/Projects/Networks/Dynamics/Simulations/cit-HepPh_gd/"
        ALL_TIME_INDEXES,DYNAMICS_PARAMETERS,macroNumbers = gd_files_handler.gd_folder_stats(gd_directory,True)
        macro_state_identifier = "basic_stats"
        macrostate_file_indentifier = "basicstatsMacro"
        #macro_keys = ["number_of_nodes","number_of_edges"]
        #df = MacrostatesHandlers.TS_dict_macro(gd_directory,
        #                                       macro_state_identifier,
        #                                       macrostate_file_indentifier,
        #                                       macro_keys)        

        #df.plot()
        #plt.show()
        
        
if __name__ == '__main__':
    import sys;sys.argv = ['','Test.timeSeriesTest']
    unittest.main()