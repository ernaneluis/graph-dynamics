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
from graph_dynamics.networks import datatypes, communities
from time import sleep

#HERE WE CONCATENATE ALL AVAILABLE GRAPH CLASSES
graph_class_dictionary = dict(datatypes.graph_class_dictionary)
graph_class_dictionary.update(communities.graph_class_dictionary)

DYNAMICS_PARAMETERS_KEYS = ["number_of_steps","number_of_steps_in_memory","simulations_directory","dynamics_identifier","graph_class","verbose","datetime_timeseries","initial_date","DynamicsClassParameters","macrostates"]
   
def files_names(DYNAMICS_PARAMETERS,time_index,macrostate_file_indentifier=None):
    """
    Parameters
    ----------
    DYNAMICS_PARAMETERS: json iwth dynamical information
     
    time_index: int 
    
    macrostate_file_indentifier: string 
    
    Returns
    -------
    gd_directory,graph_filename,graphstate_filename,macrostate_filename
            strings with the files names for a time step in a  _gd directory 
    """ 
    dynamics_identifier = DYNAMICS_PARAMETERS["dynamics_identifier"]
    gd_directory =  DYNAMICS_PARAMETERS["simulations_directory"] + dynamics_identifier + "_gd/"
    graph_filename = "{0}_gGD_{1}_.gd".format(dynamics_identifier,time_index)
    graphstate_filename = "{0}_sGD_{1}_.gd".format(dynamics_identifier,time_index)
    
    if macrostate_file_indentifier != None:
        macrostate_filename = "{0}_mGD_{1}_{2}_.gd".format(dynamics_identifier,
                                                                               macrostate_file_indentifier,                    
                                                                               time_index)
        
    return  gd_directory,graph_filename,graphstate_filename,macrostate_filename

class GraphsDynamics(object):
    """
    This is a class to specify a graph 
    
    :math:`[\mathbf{G}_1,\mathbf{G}_2,\dots,\mathbf{G}_T]`
    
    """
    __metaclass__ = ABCMeta
    def __init__(self,DYNAMICS_PARAMETERS):
        """
        Parameters
        ----------
            DYNAMICS_PARAMETERS: JSON
                this is a json object which contains all the information regarding how the dynaimcsis going to be handle:
                
                DYNAMICS_PARAMETERS = {"number_of_steps":int,
                                           "number_of_steps_in_memory":int,
                                           "simulations_directory":string,
                                           "dynamics_identifier":"string",
                                           "graph_class":string,
                                           "verbose":bool}
                                           
                number_of_steps: number of full simulations steps
                number_of_steps_in_memory: number of graphs and macro states to be kept in memory prior to ouput
                simulations_directory: where the _gd  folder is created
                graph_class: this is the class available for simulation in graph_class_dictionary 
                verbose: level of logger
                
        """ 
        self.dynamics_identifier = DYNAMICS_PARAMETERS["dynamics_identifier"]
        self.gd_directory =  DYNAMICS_PARAMETERS["simulations_directory"] + self.dynamics_identifier + "_gd/"
        input_json = set(DYNAMICS_PARAMETERS.keys())
        expected = set(DYNAMICS_PARAMETERS_KEYS)
        
        if not (input_json == expected):
            print "Wrong dynamical parameters in Dynamic Class"
            print input_json.difference(expected)
            raise Exception
        
        #make sure folder for output is ready
        if not os.path.exists(self.gd_directory):
            print "New Dynamics Directory"
            os.makedirs(self.gd_directory)
        else:
            print "Dynamics Directory Exists"
        
        json.dump(DYNAMICS_PARAMETERS,
                  open(self.gd_directory+"DYNAMICS_PARAMETERS","w"))
        
    @abstractmethod
    def generate_graphs_paths(self,initial_graph,number_of_steps):
        """
        Simulation (Prediction)
        
        This function should return a list of graphs GRAPHS_LIST where
        GRAPH_LIST[0] = INITIAL GRAPH
        len(GRAPH_LIST) = number_of_steps
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
            initial_graph: Graph Object
        """
        DYNAMICS_PARAMETERS = self.get_dynamics_state()
        steps_in_memory = DYNAMICS_PARAMETERS["number_of_steps_in_memory"]
        number_of_steps = DYNAMICS_PARAMETERS["number_of_steps"]
        macrostates_names = DYNAMICS_PARAMETERS["macrostates"]
              
        #==================================================
        # CHECK ALL FILES
        #==================================================
        ALL_DYNAMIC_FILES_NAME, GRAPH_FILES, STATE_FILES, ALL_TIME_INDEXES, latest_index = self.handle_files()
        self.latest_index = latest_index
        if self.latest_index + N + 1 >= number_of_steps:
            N = number_of_steps
        else:
            N = self.latest_index + N 
        #==================================================
        # DEFINE INITIAL GRAPH FROM LATEST STATE
        #==================================================        
        if len(GRAPH_FILES) > 0:
            initial_graph = self.get_graph(self.latest_index)
        elif initial_graph == None:
            print "Wrong graph initialization in evolve function"
            raise Exception
        print "#{0} STEPS EVOLUTION STARTED FOR {1}".format(N,self.dynamics_identifier)
        print "#STARTING EVOLUTION AT STEP {0}".format(self.latest_index)
        print "Number of initial nodes: ",initial_graph.get_networkx().number_of_nodes()
        
        if  self.latest_index <  N:
            N = N - self.latest_index
            if N < steps_in_memory:
                if self.latest_index == 0:
                    GRAPHS_IN_MEMORY = self.generate_graphs_paths(initial_graph,N)
                else:
                    GRAPHS_IN_MEMORY = self.generate_graphs_paths(initial_graph,N+1)[1:] #CHECK
                #FOR ALL GRAPHS IN MEMORY EVALUATE THE MACROSTATES AND OUTPUT
                for graph_object in GRAPHS_IN_MEMORY:
                    self.output_graph_state(graph_object,self.latest_index)
                    self.calculate_output_macrostates(graph_object,self.latest_index,macrostates_names)
                    self.latest_index += 1
            else:
                if (N % steps_in_memory) != 0:
                    steps = np.concatenate([np.repeat(steps_in_memory,N / steps_in_memory),np.array([N % steps_in_memory])])
                else:
                    steps = np.repeat(steps_in_memory,N / steps_in_memory)
                    
                for i_number_of_steps in steps:
                    if self.latest_index == 0:
                        GRAPHS_IN_MEMORY = self.generate_graphs_paths(initial_graph,i_number_of_steps)
                    else:
                        GRAPHS_IN_MEMORY = self.generate_graphs_paths(initial_graph,i_number_of_steps+1)[1:]
                    #FOR ALL GRAPHS IN MEMORY EVALUATE THE MACROSTATES AND OUTPUT
                    for  graph_object in GRAPHS_IN_MEMORY:
                        self.output_graph_state(graph_object,self.latest_index)
                        self.calculate_output_macrostates(graph_object,self.latest_index,macrostates_names)
                        self.latest_index += 1
                    
                    print "All graph in memory"
                    for graph in GRAPHS_IN_MEMORY:
                        print graph.get_networkx().number_of_nodes()
                    
                    print "last guy"    
                    latest_graph_state = GRAPHS_IN_MEMORY[-1].get_graph_state()
                    latest_graph = GRAPHS_IN_MEMORY[-1].get_networkx()
                    initial_graph = graph_class_dictionary[DYNAMICS_PARAMETERS["graph_class"]](graph_state=latest_graph_state,
                                                                                              networkx_graph=latest_graph)
                    print initial_graph.get_networkx().number_of_nodes()
                    
        else:
            print "#EVOLUTION READY"
    
    def get_graph_path_window(self,index_0,index_f):
        """
        """
        ALL_DYNAMIC_FILES_NAME, GRAPH_FILES, STATE_FILES, ALL_TIME_INDEXES, latest_index = self.handle_files()
        GRAPHS_OBJECTS = []
        if index_f in ALL_TIME_INDEXES:
            for time_index in range(index_0,index_f):
                GRAPHS_OBJECTS.append(self.get_graph(time_index))
        else:
            print "No graph found"
            return Exception
        
        return GRAPHS_OBJECTS
        
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
        ALL_DYNAMIC_FILES_NAME = os.listdir(self.gd_directory)
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
            print "#CHECK FOLDER {0}".format(self.gd_directory)
            raise Exception
        
        return  ALL_DYNAMIC_FILES_NAME, GRAPH_FILES, STATE_FILES, ALL_TIME_INDEXES, latest_index
    
    def output_graph_state(self,graph_object,latest_index):
        """
        Handles the json output
        """
        graph_state = graph_object.get_graph_state()
        graph_filename = self.gd_directory+"{0}_gGD_{1}_.gd".format(self.dynamics_identifier,latest_index)
        graphstate_filename = self.gd_directory+"{0}_sGD_{1}_.gd".format(self.dynamics_identifier,latest_index)
        #TO DO: create the edge list file without the need for networkx
        nx.write_edgelist(graph_object.get_networkx(),graph_filename)
        with open(graphstate_filename,"w") as outfile:
            json.dump(graph_state, outfile)
    
    def calculate_output_macrostates(self,graph_object,latest_index,macrostates_names):
        """
        Calculates the macro states and outputs them in folder
        """
        macrostate_filename = self.gd_directory+"{0}_mGD_{1}_{2}_.gd".format(self.dynamics_identifier,
                                                                                    self.dynamics_identifier+"-macros",                    
                                                                                    latest_index)
        #TO DO: parallelize calls to macrostates
        macrostate_json = {}
        for macrostate_function in macrostates_names:
            macrostate_function_name = macrostate_function[0]
            macrostate_function_parameters = macrostate_function[1]
            macrostate_json[macrostate_function_name] = Macrostates.macrostate_function_dictionary[macrostate_function_name](graph_object,*macrostate_function_parameters)                     
        with open(macrostate_filename,"w") as outfile:
            json.dump(macrostate_json, outfile)
            
    def get_graph(self,time_index):
        """
        Create a graph object from a given time index
        
        It only requires the state and adjacency to be defined there
        
        Returns
        -------
            graph_object: Graph object (graph_dynamics.networks.datatypes)
        """
        gd_dynamical_parameters = self.get_dynamics_state()
        graph_filename = self.gd_directory+"{0}_gGD_{1}_.gd".format(self.dynamics_identifier,time_index)
        graphstate_filename = self.gd_directory+"{0}_sGD_{1}_.gd".format(self.dynamics_identifier,time_index)
        
        latest_graph_state = json.load(open(graphstate_filename,"r"))
        latest_graph = nx.read_edgelist(graph_filename)
        
        graph_object = graph_class_dictionary[gd_dynamical_parameters["graph_class"]](graph_state=latest_graph_state,
                                                                                      networkx_graph=latest_graph)
        
        return graph_object 