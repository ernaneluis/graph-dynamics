'''
Created on Jun 9, 2017

@author: cesar
'''
import os
import copy
import json
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
from abc import ABCMeta, abstractmethod
from graph_dynamics.dynamics import Macrostates
from graph_dynamics.networks import datatypes 
from time import sleep

#HERE WE CONCATENATE ALL AVAILABLE GRAPH CLASSES
graph_class_dictionary = datatypes.graph_class_dictionary   

class GraphsDynamics(object):
    
    __metaclass__ = ABCMeta
    def __init__(self,gd_dynamical_parameters):
        """
        Parameters
        ----------
            initial_networks: Graph object (graph_dynamics.networks.datatypes)
            dynamics_identifier: string
            
        """ 
        self.dynamics_identifier = gd_dynamical_parameters["dynamics_identifier"]
        self.dynamics_foldername =  gd_dynamical_parameters["gd_directory"] + self.dynamics_identifier + "_gd/"
        #make sure folder for output is ready
        if not os.path.exists(self.dynamics_foldername):
            print "New Dynamics Directory"
            os.makedirs(self.dynamics_foldername)
        else:
            print "Dynamics Directory Exists"
            
    @abstractmethod
    def generate_graphs_paths(self,initial_graph,N):
        """
        Simulation (Prediction)   
        """
        raise NotImplemented()
    
    @abstractmethod
    def set_graph_path(self):
        """
        Empirical Data
        """
        raise NotImplemented()
        
    @abstractmethod
    def inference_on_graphs_paths(self):
        """
        Learning/Training
        """
        raise NotImplemented()
        
    @abstractmethod
    def get_dynamics_state(self):
        raise NotImplemented()
    
    def evolve(self,N,initial_graph=None):
        """
        Function for handling the evolution, files and macro states
        
        If the graph has a temporal representation, the file 
        dynamics_identifier_gGD_{0}_.gd corresponds only to the edges created 
        at that time.
        
        PARAMETERS:
        ----------
            N: int 
                number of steps
            gd_dynamical_parameters: dict
        """
        gd_dynamical_parameters = self.get_dynamics_state()
        steps_in_memory = gd_dynamical_parameters["number_of_steps_in_memory"]
        macrostates_names = gd_dynamical_parameters["macrostates"]        
        
        #==================================================
        # CHECK ALL FILES
        #==================================================
        
        ALL_DYNAMIC_FILES_NAME, GRAPH_FILES, STATE_FILES, ALL_TIME_INDEXES, latest_index = self.handle_files()
        
        #==================================================
        # DEFINE INITIAL GRAPH FROM LATEST STATE
        #==================================================        
        
        if len(GRAPH_FILES) > 0:
            initial_graph = self.get_graph(latest_index)


        print "#{0} STEPS EVOLUTION STARTED FOR {1}".format(N,self.dynamics_identifier)
        print "#STARTING EVOLUTION AT STEP {0}".format(latest_index)
        
        if  latest_index <  N:
            N = N - latest_index
            if N < steps_in_memory:
                if latest_index == 0:
                    GRAPHS_IN_MEMORY = self.generate_graphs_paths(initial_graph,N)
                else:
                    GRAPHS_IN_MEMORY = self.generate_graphs_paths(initial_graph,N)[1:]
                    
                #FOR ALL GRAPHS IN MEMORY EVALUATE THE MACROSTATES AND OUTPUT
                for graph_object in GRAPHS_IN_MEMORY:
                    self.output_graph_state(graph_object,latest_index)
                    self.calculate_output_macrostates(graph_object,latest_index,macrostates_names)
                    latest_index += 1
                    
            else:
                if (N % steps_in_memory) != 0:
                    steps = np.concatenate([np.repeat(steps_in_memory,N / steps_in_memory),np.array([N % steps_in_memory])])
                else:
                    steps = np.repeat(steps_in_memory,N / steps_in_memory)
                    
                for i_number_of_steps in steps:
                    if latest_index == 0:
                        GRAPHS_IN_MEMORY = self.generate_graphs_paths(initial_graph,i_number_of_steps)
                    else:
                        GRAPHS_IN_MEMORY = self.generate_graphs_paths(initial_graph,i_number_of_steps+1)[1:]
                    #FOR ALL GRAPHS IN MEMORY EVALUATE THE MACROSTATES AND OUTPUT
                    for  graph_object in GRAPHS_IN_MEMORY:
                        self.output_graph_state(graph_object,latest_index)
                        self.calculate_output_macrostates(graph_object,latest_index,macrostates_names)
                        latest_index += 1
                        
                initial_graph = copy.deepcopy(GRAPHS_IN_MEMORY[-1])
        else:
            print "#EVOLUTION READY"
    
    def get_graph_path_window(self,index_0,index_f):
        """
        """
        ALL_DYNAMIC_FILES_NAME, GRAPH_FILES, STATE_FILES, ALL_TIME_INDEXES, latest_index = self.handle_files()
        
        return None
        
    def handle_files(self):
        """
        Using the dynamical state we check all files
        
        Returns
        -------
        ALL_DYNAMIC_FILES_NAME, GRAPH_FILES, STATE_FILES, ALL_TIME_INDEXES, latest_index
        """
        #==================================================
        # CHECK ALL FILES
        #==================================================            
        ALL_DYNAMIC_FILES_NAME = os.listdir(self.dynamics_foldername)
        GRAPH_FILES = [filename for filename in ALL_DYNAMIC_FILES_NAME if "gGD" in filename]
        STATE_FILES = [filename for filename in ALL_DYNAMIC_FILES_NAME if "sGD" in filename]
        try:
            ALL_TIME_INDEXES = [int(filename.split("_")[2]) for filename in GRAPH_FILES]
        except:
            print "Wrong filename, dynamic identifier cannot have underscore (_) "
            raise Exception
        try:
            latest_index = max(ALL_TIME_INDEXES)    
        except:
            latest_index = 0
        
        if not (len(GRAPH_FILES) == len(STATE_FILES)):
            print "#PROBLEM WITH DYNAMICAL GRAPH FILES, FILE MISSING"
            print "#CHECK FOLDER {0}".format(self.dynamics_foldername)
            raise Exception
        
        return  ALL_DYNAMIC_FILES_NAME, GRAPH_FILES, STATE_FILES, ALL_TIME_INDEXES, latest_index
    
    def output_graph_state(self,graph_object,latest_index):
        """
        Handles the json output
        """
        graph_state = graph_object.get_graph_state()
        graph_filename = self.dynamics_foldername+"{0}_gGD_{1}_.gd".format(self.dynamics_identifier,latest_index)
        graphstate_filename = self.dynamics_foldername+"{0}_sGD_{1}_.gd".format(self.dynamics_identifier,latest_index)
        #TO DO: create the edge list file without the need for networkx
        nx.write_edgelist(graph_object.get_networkx(),graph_filename)
        with open(graphstate_filename,"w") as outfile:
            json.dump(graph_state, outfile)
    
    def calculate_output_macrostates(self,graph_object,latest_index,macrostates_names):
        """
        Calculates the macro states and outputs them in folder
        """
        macrostate_filename = self.dynamics_foldername+"{0}_mGD_{1}_.gd".format(self.dynamics_identifier,latest_index)
        #TO DO: parallelize calls to macrostates
        macrostate_json = {}
        for macrostate_function_name in macrostates_names:
            macrostate_json[macrostate_function_name] = Macrostates.macrostate_function_dictionary[macrostate_function_name](graph_object)                        
        with open(macrostate_filename,"w") as outfile:
            json.dump(macrostate_json, outfile)
            
    def get_graph(self,time_index):
        gd_dynamical_parameters = self.get_dynamics_state()
        graph_filename = self.dynamics_foldername+"{0}_gGD_{1}_.gd".format(self.dynamics_identifier,time_index)
        graphstate_filename = self.dynamics_foldername+"{0}_sGD_{1}_.gd".format(self.dynamics_identifier,time_index)
        
        latest_graph_state = json.load(open(graphstate_filename,"r"))
        latest_graph = nx.read_edgelist(graph_filename)
        
        graph_object = graph_class_dictionary[gd_dynamical_parameters["graph_class"]](graph_state=latest_graph_state,networkx_graph=latest_graph)
        return graph_object 