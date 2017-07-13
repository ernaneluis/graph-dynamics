'''
Created on Jul 10, 2017

@author: cesar
'''

import os
import sys
import numpy as np
import copy as copy
import networkx as nx

from matplotlib import pyplot as plt
from graph_dynamics.dynamics.datatypes import GraphsDynamics
from graph_dynamics.networks.datatypes import CryptocurrencyGraphs

import matplotlib 

matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['text.usetex'] = True 


class CryptoCurrencyEcosystemDynamics(GraphsDynamics):
    
    def __init__(self,identifier_string,graph_files_folder,graph_file_string,data_file_string):
        
        self.name_string = "CryptoCurrenciesDynamics"
        self.data_file_string = data_file_string
        self.identifier_string = identifier_string
        self.graph_file_string = graph_file_string
        self.graph_files_folder = graph_files_folder
        
        self.time_indexes = map(int,[filename.split("_")[3] for filename in os.listdir(self.graph_files_folder) if "data" in filename])
        self.min_index = min(self.time_indexes)
        self.max_index = max(self.time_indexes)
        
        if len(self.time_indexes) != len(range(self.min_index,self.max_index)):
            print "#====================================================================="
            print "#             WARNING !!!!!!!!!                                       "  
            print "#   File Dynamic incomplete                                           "
            print "#   Check file convention  no data in filename string                 "
            print "#                                                                     "
            print "#file: ",self.time_indexes
            print "#required: ",range(self.min_index,self.max_index)
            print "#====================================================================="
            raise Exception
        nxg = CryptocurrencyGraphs(identifier_string,
                                    graph_files_folder,
                                    graph_file_string,
                                    data_file_string,
                                    self.min_index).networkx_graph
            
        GraphsDynamics.__init__(self,nxg,"snap_shot", None)
        
    def generate_graphs_paths(self,number_of_steps=None,output_type="networkx_list",keep_path_in_memory=True):
        """
        """
        if output_type=="networkx_list":
            if keep_path_in_memory:
                self.networkx_graph_paths = [CryptocurrencyGraphs(self.identifier_string,
                                             self.graph_files_folder,
                                             self.graph_file_string,
                                             self.data_file_string,
                                             time_index).networkx_graph for time_index in self.time_indexes]
                print "Graps Paths {0} in memory".format(self.identifier_string)
                return self.networkx_graph_paths 
            else:  
                return [CryptocurrencyGraphs(self.identifier_string,
                                             self.graph_files_folder,
                                             self.graph_file_string,
                                             self.data_file_string,
                                             time_index).networkx_graph for time_index in self.time_indexes]
        else:
            print "Function not designed"
            raise NotImplemented()

    def inference_on_graphs_paths(self,graphs_paths,output_type,dynamical_process=None):
        """
        """
        print "Function not implemented"
        return None