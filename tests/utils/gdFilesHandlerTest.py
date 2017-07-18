'''
Created on Jul 18, 2017

@author: cesar
'''
import json
import unittest
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from graph_dynamics.utils import gd_files_handler

class Test(unittest.TestCase):
    
    def copyFoldersTest(self):
        """
        """
        gd_directory = "/home/cesar/Desktop/Doctorado/Projects/Networks/Dynamics/Simulations/"
        old_dynamic_identifier = "palladynamic"
        new_dynamic_identifier = "Empy-States"
        gd_files_handler.copy_and_rename_graphs(gd_directory, 
                                                old_dynamic_identifier, 
                                                new_dynamic_identifier)

        
if __name__ == '__main__':
    import sys;sys.argv = ['','Test.copyFoldersTest']
    unittest.main()