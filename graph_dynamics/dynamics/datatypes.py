'''
Created on Jun 9, 2017

@author: cesar
'''
import os
import numpy as np
from matplotlib import pyplot as plt
from abc import ABCMeta, abstractmethod
from graph_dynamics.dynamics import Macrostates
from graph_dynamics.networks import datatypes 

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
    
    def evolve(self,N):
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
        gd_directory = gd_dynamical_parameters["gd_directory"]
        steps_in_memory = gd_dynamical_parameters["number_of_steps_in_memory"]
        macrostates_names = gd_dynamical_parameters["macrostates"]
        
        self.foldername =  gd_directory + self.dynamics_identifier + "_gd/"
        if not os.path.exists(self.foldername):
            os.makedirs(self.foldername)
            
        ALL_DYNAMIC_FILES_NAME = os.listdir(self.foldername)
        GRAPH_FILES = [filename for filename in ALL_DYNAMIC_FILES_NAME if "gGD" in filename]
        STATE_FILES = [filename for filename in ALL_DYNAMIC_FILES_NAME if "sGD" in filename]
        
        ALL_TIME_INDEXES = [int(filename.split("_")[3]) for filename in GRAPH_FILES]
        try:
            latest_index = max(ALL_TIME_INDEXES)
        except:
            latest_index = 0
        
        if not (len(GRAPH_FILES) == len(STATE_FILES)):
            print "#PROBLEM WITH DYNAMICAL GRAPH FILES, FILE MISSING"
            print "#CHECK FOLDER {0}".format(self.foldername)
            raise Exception
        
        #==================================================
        # DEFINE INITIAL GRAPH FROM LATEST STATE
        #==================================================
        
        latest_graph_state_filename = "{0}_sGD_{1}_.gd".format(self.dynamics_identifier,latest_index)
        latest_graph_state = open(latest_graph_state_filename,"r").read()
        initial_graph = graph_class_dictionary[gd_dynamical_parameters["graph_class"]](latest_graph_state)
        
        #==================================================
        
        print "#{0} STEPS EVOLUTION STARTED FOR {1}".format(N,self.dynamics_identifier)
        print "#STARTING EVOLUTION AT STEP {0}".format(latest_index)
        
        if N < steps_in_memory:
            GRAPHS_IN_MEMORY = self.generate_graphs_paths(initial_graph,N)
            
            #FOR ALL GRAPHS IN MEMORY EVALUATE THE MACROSTATES AND OUTPUT
            for time_increment, graph_object in enumerate(GRAPHS_IN_MEMORY):
                edge_list = graph_object.get_edge_list()
                graph_state = graph_object.get_graph_state()
                
                latest_index = time_increment + 1
                graph_filename = self.foldername+"{0}_gGD_{1}_.gd".format(self.dynamics_identifier,latest_index)
                graphstate_filename = self.foldername+"{0}_sGD_{1}_.gd".format(self.dynamics_identifier,latest_index)
                macrostate_filename = self.foldername+"{0}_mGD_{1}_.gd".format(self.dynamics_identifier,latest_index)
                
                open(graph_filename,"w").write("\n".join(["{0} {1}".format(a[0],a[1]) for a in edge_list]))
                open(graphstate_filename,"w").write(graph_state)
                
                #TO DO: parallelize calls to macrostates
                macrostate_json = {}
                for macrostate_function_name in macrostates_names:
                    macrostate_json[macrostate_function_name] = Macrostates.macrostate_function_dictionary[macrostate_function_name](graph_object)
                open(macrostate_filename,"w").write(macrostate_json)
        else:
            steps = np.concatenate([np.repeat(steps_in_memory,N / steps_in_memory),np.array([N % steps_in_memory])])
            for i_number_of_steps in steps:
                
                GRAPHS_IN_MEMORY = self.generate_graphs_paths(i_number_of_steps)        
                #FOR ALL GRAPHS IN MEMORY EVALUATE THE MACROSTATES AND OUTPUT
                for time_increment, graph_object in enumerate(GRAPHS_IN_MEMORY):
                    edge_list = graph_object.get_edge_list()
                    graph_state = graph_object.get_graph_state()
                    
                    latest_index = time_increment + 1
                    graph_filename = self.foldername+"{0}_gGD_{1}_.gd".format(self.dynamics_identifier,latest_index)
                    graphstate_filename = self.foldername+"{0}_sGD_{1}_.gd".format(self.dynamics_identifier,latest_index)
                    macrostate_filename = self.foldername+"{0}_mGD_{1}_.gd".format(self.dynamics_identifier,latest_index)
                    
                    #TO DO: create the edge list file without the need for networkx
                    graph_object.get_networkx().write_edgelist(graph_object,graph_filename)
                    open(graphstate_filename,"w").write(graph_state)
                    
                    #TO DO: parallelize calls to macrostates
                    macrostate_json = {}
                    for macrostate_function_name in macrostates_names:
                        macrostate_json[macrostate_function_name] = Macrostates.macrostate_function_dictionary[macrostate_function_name](graph_object)
                    open(macrostate_filename,"w").write(macrostate_json)
                    