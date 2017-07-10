'''
Created on Jul 10, 2017

@author: cesar
'''
import json
import unittest
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from graph_dynamics.networks.datatypes import CryptocurrencyGraphs

class Test(unittest.TestCase):
    
    def generateCryptocurrencyGraph(self):
        time_index = 20
        identifier_string = "Crash Detections (Data up to 24/06/17)"
        graph_files_folder = "/home/cesar/Desktop/Doctorado/Projects/Networks/Cryptocurrencies/Results/cryptocurrencies_graphdynamics/"
        data_file_string = 'cryptocurrencies_graphdynamics_snapshot_{0}_data.cpickle3'
        graph_file_string = 'cryptocurrencies_graphdynamics_snapshot_{0}_graph.txt'
        CryptocurrencyEcosystem = CryptocurrencyGraphs(identifier_string,
                                                       graph_files_folder,
                                                       graph_file_string,
                                                       data_file_string,
                                                       time_index)
        print CryptocurrencyEcosystem.ecosystem_crash()
        
        
if __name__ == '__main__':
    import sys;sys.argv = ['','Test.generateCryptocurrencyGraph']
    unittest.main()